// State
let authHeader = '';
let ws = null;
let modelCatalogLoaded = false;
let currentMetricsPeriod = '1h';
let currentTheme = 'dark';

// Request details cache for inspector
let requestDetailsCache = {};

function escapeHtml(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check for existing auth
    const savedAuth = localStorage.getItem('ccnim_admin_auth');
    if (savedAuth) {
        authHeader = savedAuth;
    } else {
        authHeader = '';
    }
    
    // Check for saved theme
    const savedTheme = localStorage.getItem('ccnim_theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        applyTheme(currentTheme);
    }
    
    // Backend currently allows open admin routes; don't block UI behind login.
    showApp();
    
    // Setup keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
});

function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    authHeader = username || password ? ('Basic ' + btoa(username + ':' + password)) : '';
    localStorage.setItem('ccnim_admin_auth', authHeader);
    showApp();
}

function showApp() {
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('app').style.display = 'flex';
    
    // Connect WebSocket
    connectWebSocket();
    
    // Initial data fetch
    loadDashboard();
    loadLogs();
    loadErrors();
    loadModelPerformance();
    loadKeys();
    loadHealth();
    refreshModelCatalog();
    loadMetrics('1h');
    
    // Refresh interval
    setInterval(loadDashboard, 5000);
    setInterval(loadLogs, 5000);
    setInterval(loadErrors, 10000);
    setInterval(loadModelPerformance, 15000);
    setInterval(loadKeys, 5000);
    setInterval(loadHealth, 10000);
}

// Keyboard Shortcuts
function handleKeyboardShortcuts(e) {
    // Ignore if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    // g + d = go to dashboard
    if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
        // Wait for next key
        document.addEventListener('keydown', function nextKey(e2) {
            document.removeEventListener('keydown', nextKey);
            if (e2.key === 'd') showTab('dashboard');
            else if (e2.key === 'l') showTab('logs');
            else if (e2.key === 'e') showTab('errors');
            else if (e2.key === 'm') showTab('models');
            else if (e2.key === 'k') showTab('keys');
            else if (e2.key === 'h') showTab('health');
        });
    }
    
    // t = toggle theme
    if (e.key === 't') {
        toggleTheme();
    }
    
    // r = refresh
    if (e.key === 'r') {
        loadDashboard();
        loadLogs();
    }
    
    // Escape = close modals
    if (e.key === 'Escape') {
        closeInspector();
        closeSettingsModal();
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/admin/ws`;
    
    // Get auth from current connection
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log' && data.data) {
            addLogEntry(data.data);
            updateDashboardStats();
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Dashboard
function loadDashboard() {
    fetch('/admin/status', { headers: { 'Authorization': authHeader } })
        .then(res => {
            if (!res.ok) {
                throw new Error(`status ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            updateDashboard(data);
        })
        .catch(err => {
            console.error('Error loading dashboard:', err);
            const statusEl = document.getElementById('model-update-status');
            if (statusEl) {
                statusEl.textContent = 'Dashboard data unavailable. Check server logs.';
                statusEl.className = 'model-status error';
            }
        });
}

function updateDashboard(data) {
    // Update header stats
    const agg = data.aggregate || {};
    document.getElementById('rpm-stat').textContent = `RPM: ${agg.requests_per_minute || 0}`;
    document.getElementById('success-rate-stat').textContent = `Success: ${agg.success_rate || 100}%`;
    document.getElementById('total-requests-stat').textContent = `Total: ${agg.total_requests || 0}`;
    
    // Active model
    const modelView = document.getElementById('active-model-view');
    const activeModel = data.active_model || 'unknown';
    const defaultModel = data.default_model || 'unknown';
    const hasOverride = !!data.has_runtime_model_override;
    modelView.innerHTML = `
        <div class="model-item primary">
            <span class="model-name">${activeModel}</span>
            <span class="model-pos">${hasOverride ? 'runtime override' : 'from .env MODEL'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Default</span>
            <span class="settings-value">${defaultModel}</span>
        </div>
    `;
    const modelInput = document.getElementById('active-model-input');
    if (modelInput && document.activeElement !== modelInput) {
        modelInput.value = activeModel;
    }
    const statusEl = document.getElementById('model-update-status');
    if (statusEl && data.provider_error) {
        statusEl.textContent = `Provider unavailable: ${data.provider_error}`;
        statusEl.className = 'model-status error';
    }
    if (!modelCatalogLoaded) {
        refreshModelCatalog();
    }
    
    // Fallback summary
    const fallbackInfo = document.getElementById('fallback-info');
    const blockedKeys = (data.keys || []).filter(k => k.blocked).length;
    const fallbackModels = (data.fallback_models || []);
    fallbackInfo.innerHTML = `
        <div class="settings-item">
            <span class="settings-label">Strategy</span>
            <span class="settings-value">Model fallback + API key rotation</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Fallback Order</span>
            <span class="settings-value">${fallbackModels.length || 1} models</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Total Keys</span>
            <span class="settings-value">${(data.keys || []).length}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Blocked Keys</span>
            <span class="settings-value">${blockedKeys}</span>
        </div>
    `;
    fallbackModels.forEach((model, idx) => {
        const div = document.createElement('div');
        div.className = idx === 0 ? 'model-item primary' : 'model-item fallback';
        div.innerHTML = `
            <span class="model-name">${escapeHtml(model)}</span>
            <span class="model-pos">${idx === 0 ? 'primary' : `fallback ${idx}`}</span>
        `;
        fallbackInfo.appendChild(div);
    });

    const quickModelsEl = document.getElementById('quick-models');
    const quickSwitchModels = data.quick_switch_models || [];
    if (quickModelsEl) {
        quickModelsEl.innerHTML = '';
        if (quickSwitchModels.length > 0) {
            const title = document.createElement('div');
            title.className = 'quick-models-title';
            title.textContent = 'Quick Switch';
            quickModelsEl.appendChild(title);

            quickSwitchModels.forEach((model) => {
                const btn = document.createElement('button');
                btn.className = 'action-btn quick-model-btn';
                btn.textContent = model;
                btn.disabled = model === activeModel;
                btn.onclick = () => switchQuickModel(model);
                quickModelsEl.appendChild(btn);
            });
        }
    }
    
    // Keys Summary
    const keysSummary = document.getElementById('keys-summary');
    keysSummary.innerHTML = '';
    data.keys.forEach(key => {
        const div = document.createElement('div');
        div.className = 'keys-summary-item';
        div.innerHTML = `
            <span class="key-badge">${key.key_masked || ('***' + key.key_suffix)}</span>
            <span class="${key.blocked ? 'status-blocked' : 'status-active'}">
                ${key.blocked ? `🔴 Blocked (${key.cooldown_remaining_seconds || 0}s)` : `🟢 ${key.remaining_requests || 0} left`}
            </span>
        `;
        keysSummary.appendChild(div);
    });
    
    // Settings
    const settingsInfo = document.getElementById('settings-info');
    settingsInfo.innerHTML = `
        <div class="settings-item">
            <span class="settings-label">Rate Limit</span>
            <span class="settings-value">${data.settings.rate_limit} req/${data.settings.rate_window}s</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Cooldown</span>
            <span class="settings-value">${data.settings.cooldown_seconds}s</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Max In Flight</span>
            <span class="settings-value">${data.settings.max_in_flight || '-'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Request Timeout</span>
            <span class="settings-value">${data.settings.request_timeout_sec || '-'}s</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">OpenAI Retries</span>
            <span class="settings-value">${data.settings.openai_max_retries ?? '-'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Hard Max Tokens</span>
            <span class="settings-value">${data.settings.hard_max_tokens || '-'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Active Requests</span>
            <span class="settings-value">${(data.runtime && data.runtime.active_requests) || 0}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">Sticky Model</span>
            <span class="settings-value">${(data.runtime && data.runtime.sticky_model) || '-'}</span>
        </div>
        <div class="settings-item">
            <span class="settings-label">HTTP Latency</span>
            <span class="settings-value">avg ${agg.avg_http_latency_ms || 0}ms / p95 ${agg.p95_http_latency_ms || 0}ms</span>
        </div>
    `;
}

function updateDashboardStats() {
    // Small update when new log comes in
    const rpmStat = document.getElementById('rpm-stat');
    const currentRPM = parseInt(rpmStat.textContent.split(':')[1]) || 0;
    rpmStat.textContent = `RPM: ${currentRPM + 1}`;
}

// Logs
function loadLogs() {
    fetch('/admin/logs?limit=200', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('logs-body');
            tbody.innerHTML = '';
            data.logs.forEach(log => addLogEntry(log));
            updateLogCount();
        })
        .catch(err => console.error('Error loading logs:', err));
}

function addLogEntry(log) {
    const tbody = document.getElementById('logs-body');
    const tr = document.createElement('tr');
    
    const time = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '';
    const statusClass = log.status === 'success'
        ? 'status-success'
        : (log.status === 'fallback' ? 'status-fallback' : 'status-error');
    
    tr.innerHTML = `
        <td class="log-time">${time}</td>
        <td><code>${log.model}</code></td>
        <td><code>***${log.key_suffix}</code></td>
        <td class="${statusClass}" title="${escapeHtml(log.error || '')}">${log.status}</td>
        <td>${log.response_time_ms ? log.response_time_ms.toFixed(0) + 'ms' : '-'}</td>
        <td><button class="action-btn" onclick="inspectLog('${log.request_id || ''}', '${escapeHtml(log.model)}', '${escapeHtml(log.key_suffix)}', '${escapeHtml(log.error || '')}')">👁</button></td>
    `;
    
    // Store in cache for inspector
    if (log.request_id) {
        requestDetailsCache[log.request_id] = log;
    }
    
    tbody.insertBefore(tr, tbody.firstChild);
    
    // Keep only last 200 in DOM
    while (tbody.children.length > 200) {
        tbody.removeChild(tbody.lastChild);
    }
}

function inspectLog(requestId, model, keySuffix, error) {
    // Try to get from cache first, then fetch
    if (requestId && requestDetailsCache[requestId]) {
        showInspector(requestDetailsCache[requestId]);
    } else {
        // Show basic info if no detailed data
        showInspector({
            request_id: requestId || 'N/A',
            model: model,
            key_suffix: keySuffix,
            timestamp: new Date().toISOString(),
            error: error,
            response_body: '(Request details not available)',
        });
    }
}

function filterLogs() {
    const modelFilter = document.getElementById('log-filter-model').value.toLowerCase();
    const keyFilter = document.getElementById('log-filter-key').value.toLowerCase();
    const statusFilter = document.getElementById('log-filter-status').value;
    
    const rows = document.querySelectorAll('#logs-body tr');
    rows.forEach(row => {
        const model = row.cells[1].textContent.toLowerCase();
        const key = row.cells[2].textContent.toLowerCase();
        const statusCell = row.querySelector('.status-success, .status-error, .status-fallback');
        const status = statusCell ? statusCell.textContent.toLowerCase() : '';
        
        const matchModel = !modelFilter || model.includes(modelFilter);
        const matchKey = !keyFilter || key.includes(keyFilter);
        const matchStatus = !statusFilter || status.includes(statusFilter);
        
        row.style.display = matchModel && matchKey && matchStatus ? '' : 'none';
    });
    
    updateLogCount();
}

function updateLogCount() {
    const visible = document.querySelectorAll('#logs-body tr:not([style*="display: none"])').length;
    document.getElementById('log-count').textContent = `${visible} logs`;
}

// Keys
function loadKeys() {
    fetch('/admin/keys', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('keys-body');
            tbody.innerHTML = '';
            data.keys.forEach(key => {
                const tr = document.createElement('tr');
                const capacityPercent = key.capacity_percent || 0;
                const capacityClass = capacityPercent > 80 ? 'danger' : (capacityPercent > 60 ? 'warning' : '');
                tr.innerHTML = `
                    <td><code>${key.key_masked}</code></td>
                    <td class="${key.blocked ? 'status-blocked' : 'status-active'}">
                        ${key.blocked ? 'Blocked' : 'Active'}
                    </td>
                    <td>${key.usage_count}/${key.in_flight + key.usage_count}</td>
                    <td>${key.in_flight}</td>
                    <td>
                        <div class="health-bar">
                            <div class="health-fill ${capacityClass}" style="width: ${capacityPercent}%"></div>
                        </div>
                        <span style="font-size: 0.75rem;">${capacityPercent}%</span>
                    </td>
                    <td>
                        ${key.blocked ? 
                            `<button class="action-btn unblock" onclick="unblockKey('${key.key_suffix}')">Unblock</button>` :
                            `<button class="action-btn block" onclick="blockKey('${key.key_suffix}')">Block</button>`
                        }
                    </td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error('Error loading keys:', err));
}

function blockKey(keySuffix) {
    if (!confirm(`Block key ending in ${keySuffix}?`)) return;
    
    fetch(`/admin/keys/${keySuffix}/block`, {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => loadKeys())
    .catch(err => console.error('Error blocking key:', err));
}

function unblockKey(keySuffix) {
    fetch(`/admin/keys/${keySuffix}/unblock`, {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => loadKeys())
    .catch(err => console.error('Error unblocking key:', err));
}

function updateActiveModel() {
    const model = document.getElementById('active-model-input').value.trim();
    const persist = !!document.getElementById('persist-model-checkbox').checked;
    const statusEl = document.getElementById('model-update-status');
    if (!model) {
        statusEl.textContent = 'Model is required';
        statusEl.className = 'model-status error';
        return;
    }

    fetch('/admin/model', {
        method: 'POST',
        headers: {
            'Authorization': authHeader,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model, persist })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'updated') {
            statusEl.textContent = data.persisted ? 'Model updated and persisted to .env' : 'Runtime model updated';
            statusEl.className = 'model-status success';
            loadDashboard();
        } else {
            statusEl.textContent = data.detail || 'Failed to update model';
            statusEl.className = 'model-status error';
        }
    })
    .catch(err => {
        console.error('Error updating model:', err);
        statusEl.textContent = 'Error updating model';
        statusEl.className = 'model-status error';
    });
}

function switchQuickModel(model) {
    const input = document.getElementById('active-model-input');
    if (input) {
        input.value = model;
    }
    updateActiveModel();
}

function resetActiveModel() {
    const statusEl = document.getElementById('model-update-status');
    fetch('/admin/model/reset', {
        method: 'POST',
        headers: { 'Authorization': authHeader }
    })
    .then(res => res.json())
    .then(() => {
        statusEl.textContent = 'Runtime override cleared';
        statusEl.className = 'model-status success';
        loadDashboard();
    })
    .catch(err => {
        console.error('Error resetting model:', err);
        statusEl.textContent = 'Error resetting model';
        statusEl.className = 'model-status error';
    });
}

function refreshModelCatalog() {
    fetch('/admin/models?limit=500', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const datalist = document.getElementById('nim-model-options');
            if (!datalist) return;
            datalist.innerHTML = '';
            (data.models || []).forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                datalist.appendChild(option);
            });
            modelCatalogLoaded = true;
        })
        .catch(err => console.error('Error loading model catalog:', err));
}

// =============================================================================
// NEW PHASE 1: Request Inspector
// =============================================================================

function inspectRequest(requestId) {
    fetch(`/admin/requests/${requestId}`, { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            if (data.request) {
                showInspector(data.request);
            }
        })
        .catch(err => console.error('Error loading request details:', err));
}

function showInspector(request) {
    const modal = document.getElementById('inspector-modal');
    document.getElementById('inspector-request').textContent = JSON.stringify({
        model: request.model,
        timestamp: request.timestamp,
        key_suffix: request.key_suffix,
    }, null, 2);
    document.getElementById('inspector-response').textContent = request.response_body || '(No response body stored)';
    document.getElementById('inspector-error').textContent = request.error || '(No error)';
    modal.classList.add('active');
}

function closeInspector() {
    document.getElementById('inspector-modal').classList.remove('active');
}

// =============================================================================
// NEW PHASE 1: Error Analysis
// =============================================================================

function loadErrors() {
    fetch('/admin/errors?limit=100', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            updateErrorsDisplay(data);
        })
        .catch(err => console.error('Error loading errors:', err));
}

function updateErrorsDisplay(data) {
    // Update category summary
    const summaryEl = document.getElementById('error-category-summary');
    summaryEl.innerHTML = '';
    const byCategory = data.by_category || {};
    const totalErrors = data.total || 0;
    
    // Add total badge
    const totalBadge = document.createElement('span');
    totalBadge.className = 'error-category-badge unknown';
    totalBadge.textContent = `Total: ${totalErrors}`;
    summaryEl.appendChild(totalBadge);
    
    // Add category badges
    Object.entries(byCategory).forEach(([category, count]) => {
        const badge = document.createElement('span');
        badge.className = `error-category-badge ${category}`;
        badge.textContent = `${category}: ${count}`;
        badge.onclick = () => {
            document.getElementById('error-filter-category').value = category;
            filterErrors();
        };
        badge.style.cursor = 'pointer';
        summaryEl.appendChild(badge);
    });
    
    // Update error count
    document.getElementById('error-count').textContent = `${totalErrors} errors`;
    
    // Populate errors table
    const tbody = document.getElementById('errors-body');
    tbody.innerHTML = '';
    
    (data.errors || []).forEach(error => {
        const tr = document.createElement('tr');
        const time = error.timestamp ? new Date(error.timestamp).toLocaleTimeString() : '';
        tr.innerHTML = `
            <td class="log-time">${time}</td>
            <td><code>${error.model}</code></td>
            <td><span class="error-category-badge ${error.category}" style="font-size: 0.75rem;">${error.category}</span></td>
            <td class="error-message" title="${escapeHtml(error.error || '')}">${error.error ? error.error.substring(0, 100) : '-'}</td>
            <td>${error.response_time_ms ? error.response_time_ms.toFixed(0) + 'ms' : '-'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function filterErrors() {
    const modelFilter = document.getElementById('error-filter-model').value.toLowerCase();
    const categoryFilter = document.getElementById('error-filter-category').value;
    
    const rows = document.querySelectorAll('#errors-body tr');
    rows.forEach(row => {
        const model = row.cells[1].textContent.toLowerCase();
        const category = row.cells[2].textContent.toLowerCase();
        
        const matchModel = !modelFilter || model.includes(modelFilter);
        const matchCategory = !categoryFilter || category.includes(categoryFilter);
        
        row.style.display = matchModel && matchCategory ? '' : 'none';
    });
    
    // Update visible count
    const visible = document.querySelectorAll('#errors-body tr:not([style*="display: none"])').length;
    document.getElementById('error-count').textContent = `${visible} errors`;
}

// =============================================================================
// NEW PHASE 1: Model Performance
// =============================================================================

function loadModelPerformance() {
    fetch('/admin/models/performance', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            updateModelPerformance(data.models || []);
        })
        .catch(err => console.error('Error loading model performance:', err));
}

function updateModelPerformance(models) {
    const container = document.getElementById('model-performance');
    if (!container) return;
    
    if (models.length === 0) {
        container.innerHTML = '<p style="color: #888;">No performance data available yet.</p>';
        return;
    }
    
    container.innerHTML = '<div class="model-performance-grid"></div>';
    const grid = container.querySelector('.model-performance-grid');
    
    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'model-performance-card';
        
        const successRateClass = model.success_rate < 50 ? 'critical' : (model.success_rate < 80 ? 'low' : '');
        
        card.innerHTML = `
            <div class="model-name">${escapeHtml(model.model)}</div>
            <div class="model-performance-stats">
                <span>Requests: <span class="stat">${model.total_requests}</span></span>
                <span>Avg Success Latency: <span class="stat">${model.avg_latency_ms}ms</span></span>
            </div>
            <div class="success-rate-bar">
                <div class="fill ${successRateClass}" style="width: ${model.success_rate}%"></div>
            </div>
            <div class="model-performance-stats">
                <span>Success: <span class="stat" style="color: #4caf50;">${model.success}</span></span>
                <span>Failed: <span class="stat" style="color: #f44336;">${model.failed}</span></span>
                <span>Fallback: <span class="stat" style="color: #ffb74d;">${model.fallback}</span></span>
            </div>
            <div class="model-performance-stats">
                <span>Avg Attempt: <span class="stat">${model.avg_attempt_latency_ms || 0}ms</span></span>
            </div>
            ${model.recent_errors && model.recent_errors.length > 0 ? 
                `<div class="model-errors">Last error: ${escapeHtml(model.recent_errors[0].substring(0, 80))}</div>` : ''}
        `;
        grid.appendChild(card);
    });
}

// =============================================================================
// NEW PHASE 1: Settings Editor
// =============================================================================

function showSettingsEditor() {
    fetch('/admin/settings', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            document.getElementById('setting-rate-limit').value = data.rate_limit;
            document.getElementById('setting-rate-window').value = data.rate_window;
            document.getElementById('setting-cooldown').value = data.cooldown_seconds;
            document.getElementById('setting-max-inflight').value = data.max_in_flight;
            document.getElementById('setting-model').value = data.model;
            document.getElementById('settings-modal').classList.add('active');
        })
        .catch(err => console.error('Error loading settings:', err));
}

function closeSettingsModal() {
    document.getElementById('settings-modal').classList.remove('active');
}

// =============================================================================
// NEW PHASE 2: Health
// =============================================================================

function loadHealth() {
    fetch('/admin/health/detailed', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            updateHealthDisplay(data);
        })
        .catch(err => console.error('Error loading health:', err));
}

function updateHealthDisplay(data) {
    // System health status
    const healthEl = document.getElementById('system-health');
    const statusClass = data.status === 'healthy' ? 'health-status-healthy' : 
                        data.status === 'degraded' ? 'health-status-degraded' : 'health-status-unhealthy';
    healthEl.innerHTML = `
        <div class="${statusClass}">
            <span style="font-size: 2rem;">${data.status === 'healthy' ? '✓' : data.status === 'degraded' ? '⚠' : '✗'}</span>
            <span style="font-size: 1.2rem; text-transform: capitalize;">${data.status}</span>
        </div>
    `;
    
    // Components
    const componentsEl = document.getElementById('component-health');
    componentsEl.innerHTML = '';
    const components = data.components || {};
    Object.entries(components).forEach(([name, info]) => {
        const compStatusClass = info.status === 'healthy' ? 'health-status-healthy' : 'health-status-unhealthy';
        const div = document.createElement('div');
        div.className = 'component-item';
        div.innerHTML = `
            <span class="component-name">${name}</span>
            <span class="${compStatusClass}">
                ${info.status === 'healthy' ? '✓' : '✗'}
            </span>
        `;
        componentsEl.appendChild(div);
    });
    
    // Warnings
    const warningsEl = document.getElementById('health-warnings');
    warningsEl.innerHTML = '';
    const warnings = data.warnings || [];
    if (warnings.length === 0) {
        warningsEl.innerHTML = '<p style="color: #4caf50;">No warnings</p>';
    } else {
        warnings.forEach(warning => {
            const div = document.createElement('div');
            div.className = 'warning-item';
            div.innerHTML = `<span>⚠</span><span>${escapeHtml(warning)}</span>`;
            warningsEl.appendChild(div);
        });
    }
}

// =============================================================================
// NEW PHASE 2: Metrics Charts (Simple bar chart without external library)
// =============================================================================

function loadMetrics(period) {
    currentMetricsPeriod = period;
    fetch(`/admin/metrics/history?period=${period}`, { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            renderMetricsChart(data.data || []);
        })
        .catch(err => console.error('Error loading metrics:', err));
    
    // Update active button
    document.querySelectorAll('.chart-controls .action-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === period) btn.classList.add('active');
    });
}

function renderMetricsChart(data) {
    const container = document.querySelector('.chart-container');
    if (!container) return;
    
    // Clear existing chart
    container.innerHTML = '';
    
    if (data.length === 0) {
        container.innerHTML = '<p style="color: #888; text-align: center; padding: 2rem;">No data available for this period</p>';
        return;
    }
    
    // Find max for scaling
    const maxRequests = Math.max(...data.map(d => d.requests), 1);
    
    // Create simple bar chart
    const chart = document.createElement('div');
    chart.style.cssText = 'display: flex; align-items: flex-end; height: 100%; gap: 2px; padding: 10px;';
    
    data.forEach(bucket => {
        const bar = document.createElement('div');
        const height = (bucket.requests / maxRequests) * 100;
        const color = bucket.failed > 0 ? '#f44336' : (bucket.fallback > 0 ? '#ffb74d' : '#4caf50');
        
        bar.style.cssText = `
            flex: 1;
            height: ${height}%;
            background: ${color};
            min-height: 2px;
            position: relative;
            border-radius: 2px 2px 0 0;
            transition: height 0.3s;
        `;
        bar.title = `${bucket.timestamp}: ${bucket.requests} requests (${bucket.success} success, ${bucket.failed} failed, ${bucket.fallback} fallback)`;
        
        chart.appendChild(bar);
    });
    
    container.appendChild(chart);
}

// =============================================================================
// NEW PHASE 3: Key Testing
// =============================================================================

function testAllKeys() {
    fetch('/admin/keys/test', { 
        method: 'POST',
        headers: { 'Authorization': authHeader } 
    })
    .then(res => res.json())
    .then(data => {
        alert(`Tested ${data.keys ? data.keys.length : 0} keys`);
        loadKeys();
    })
    .catch(err => console.error('Error testing keys:', err));
}

// =============================================================================
// NEW PHASE 4: Theme & Export
// =============================================================================

function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(currentTheme);
    localStorage.setItem('ccnim_theme', currentTheme);
}

function applyTheme(theme) {
    if (theme === 'light') {
        document.body.classList.add('light-theme');
        document.getElementById('theme-toggle').textContent = '☀️';
    } else {
        document.body.classList.remove('light-theme');
        document.getElementById('theme-toggle').textContent = '🌙';
    }
}

function exportLogs(format) {
    window.location.href = `/admin/export/logs?format=${format}`;
}

function exportMetrics() {
    fetch('/admin/export/metrics', { headers: { 'Authorization': authHeader } })
        .then(res => res.json())
        .then(data => {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ccnim-metrics.json';
            a.click();
            URL.revokeObjectURL(url);
        })
        .catch(err => console.error('Error exporting metrics:', err));
}
