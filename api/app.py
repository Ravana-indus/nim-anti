"""FastAPI application factory and configuration."""

import os

# Opt-in to future behavior for python-telegram-bot
os.environ["PTB_TIMEDELTA"] = "1"

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes import router
from .admin import router as admin_router
from .dependencies import cleanup_provider
from providers.exceptions import ProviderError
from config.settings import get_settings
from .telemetry import telemetry

# Configure logging (atomic - only on true fresh start)
_settings = get_settings()
LOG_FILE = _settings.log_file

# Check if logging is already configured (e.g., hot reload)
# If handlers exist, skip setup to avoid clearing logs mid-session
if not logging.root.handlers:
    # Fresh start - clear log file and configure
    open(LOG_FILE, "w", encoding="utf-8").close()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")],
    )

logger = logging.getLogger(__name__)

# Suppress noisy uvicorn logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("Starting Claude Code Proxy...")

    # Initialize messaging platform if configured
    messaging_platform = None
    message_handler = None
    cli_manager = None

    try:
        # Use the messaging factory to create the right platform
        from messaging.factory import create_messaging_platform

        messaging_platform = create_messaging_platform(
            platform_type=settings.messaging_platform,
            bot_token=settings.telegram_bot_token,
            allowed_user_id=settings.allowed_telegram_user_id,
        )

        if messaging_platform:
            from messaging.handler import ClaudeMessageHandler
            from messaging.session import SessionStore
            from cli.manager import CLISessionManager

            # Setup workspace - CLI runs in allowed_dir if set (e.g. project root)
            workspace = (
                os.path.abspath(settings.allowed_dir)
                if settings.allowed_dir
                else os.getcwd()
            )
            os.makedirs(workspace, exist_ok=True)

            # Session data stored in agent_workspace
            data_path = os.path.abspath(settings.claude_workspace)
            os.makedirs(data_path, exist_ok=True)

            api_url = f"http://{settings.host}:{settings.port}/v1"
            allowed_dirs = [workspace] if settings.allowed_dir else []
            cli_manager = CLISessionManager(
                workspace_path=workspace,
                api_url=api_url,
                allowed_dirs=allowed_dirs,
                max_sessions=settings.max_cli_sessions,
            )

            # Initialize session store
            session_store = SessionStore(
                storage_path=os.path.join(data_path, "sessions.json")
            )

            # Create and register message handler
            message_handler = ClaudeMessageHandler(
                platform=messaging_platform,
                cli_manager=cli_manager,
                session_store=session_store,
            )

            # Restore tree state if available
            saved_trees = session_store.get_all_trees()
            if saved_trees:
                logger.info(f"Restoring {len(saved_trees)} conversation trees...")
                from messaging.tree_queue import TreeQueueManager

                message_handler.tree_queue = TreeQueueManager.from_dict(
                    {
                        "trees": saved_trees,
                        "node_to_tree": session_store.get_node_mapping(),
                    },
                    queue_update_callback=message_handler._update_queue_positions,
                    node_started_callback=message_handler._mark_node_processing,
                )
                # Reconcile restored state - anything PENDING/IN_PROGRESS is lost across restart
                if message_handler.tree_queue.cleanup_stale_nodes() > 0:
                    # Sync back and save
                    tree_data = message_handler.tree_queue.to_dict()
                    session_store.sync_from_tree_data(
                        tree_data["trees"], tree_data["node_to_tree"]
                    )

            # Wire up the handler
            messaging_platform.on_message(message_handler.handle_message)

            # Start the platform
            await messaging_platform.start()
            logger.info(
                f"{messaging_platform.name} platform started with message handler"
            )

    except ImportError as e:
        logger.warning(f"Messaging module import error: {e}")
    except Exception as e:
        logger.error(f"Failed to start messaging platform: {e}")
        import traceback

        logger.error(traceback.format_exc())

    # Store in app state for access in routes
    app.state.messaging_platform = messaging_platform
    app.state.message_handler = message_handler
    app.state.cli_manager = cli_manager

    yield

    # Cleanup
    if messaging_platform:
        await messaging_platform.stop()
    if cli_manager:
        await cli_manager.stop_all()
    await cleanup_provider()
    logger.info("Server shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Claude Code Proxy",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)
    app.include_router(admin_router)

    # Mount admin static files
    app.mount("/admin/static", StaticFiles(directory="static/admin"), name="admin-static")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Request timeout middleware - prevents hung requests from consuming resources
    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        """Enforce request timeout to prevent hung requests."""
        settings = get_settings()
        timeout_seconds = getattr(settings, 'request_timeout_seconds', 300.0)
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {timeout_seconds}s: {request.method} {request.url.path}")
            return JSONResponse(
                status_code=504,
                content={
                    "type": "error",
                    "error": {
                        "type": "timeout_error",
                        "message": f"Request timed out after {timeout_seconds} seconds.",
                    },
                },
            )

    # Middleware for logging requests to admin dashboard
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            telemetry.record_http(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=(time.time() - start_time) * 1000,
            )

    @app.get("/metrics")
    async def metrics():
        """Prometheus-compatible metrics endpoint."""
        return Response(content=telemetry.as_prometheus(), media_type="text/plain")

    # Health check endpoints for orchestration (Kubernetes, Docker, etc.)
    @app.get("/health")
    async def health():
        """Basic health check - always returns healthy if server is running."""
        return {"status": "healthy", "service": "claude-code-proxy"}

    @app.get("/admin", include_in_schema=False)
    async def admin_redirect():
        """Redirect /admin to /admin/ for admin dashboard."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin/", status_code=301)

    @app.get("/ready")
    async def ready():
        """Readiness check - verifies critical dependencies are available."""
        checks = {
            "server": True,
            "provider": False,
            "key_manager": False,
        }
        
        try:
            # Check if provider is initialized
            from .dependencies import get_provider
            provider = get_provider()
            checks["provider"] = provider is not None
            
            # Check if key manager has available keys
            if hasattr(provider, '_key_manager'):
                available_keys = provider._key_manager.get_available_keys()
                checks["key_manager"] = len(available_keys) > 0
            else:
                checks["key_manager"] = True
        except Exception as e:
            logger.warning(f"Readiness check failed: {e}")
        
        all_ready = all(checks.values())
        status_code = 200 if all_ready else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if all_ready else "not_ready",
                "checks": checks,
            },
        )

    # Exception handlers
    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        logger.error(f"General Error: {str(exc)}")
        import traceback

        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred.",
                },
            },
        )

    return app


# Default app instance for uvicorn
app = create_app()
