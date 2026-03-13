# Admin Fallback Order Drag-and-Drop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add drag-and-drop fallback reordering in the admin dashboard and persist the configured fallback order, plus optional next-restart `MODEL`, into `.env`.

**Architecture:** Keep persistence logic centralized in `config/settings.py`, replace the existing admin placeholder endpoint with a validated save flow, and render drag/drop entirely in browser state until the user explicitly saves. The request path should continue using cached settings so the feature remains control-plane only.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, vanilla JavaScript, existing admin CSS

---

### Task 1: Lock backend persistence behavior with tests

**Files:**
- Modify: `tests/test_admin.py`
- Modify: `tests/test_model_utils.py`
- Modify: `api/admin.py`
- Modify: `config/settings.py`

**Step 1: Write the failing test**

Add tests covering:

```python
response = client.post("/admin/fallback-order", json={
    "models": ["model-c", "model-a", "model-b"],
    "persist_default_for_next_restart": True,
})
assert response.status_code == 200
```

and a unit test for the persistence helper that verifies both `NVIDIA_NIM_FALLBACK_MODELS` and optional `MODEL` updates.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_admin.py -k fallback_order tests/test_model_utils.py -k persist_fallback`
Expected: FAIL because the endpoint is still `not_implemented` and no fallback persistence helper exists.

**Step 3: Write minimal implementation**

- Add a shared `.env` update helper in `config/settings.py`
- Add a fallback persistence helper with normalization, dedupe, and optional `MODEL` update
- Replace `POST /admin/fallback-order` in `api/admin.py` with a validated implementation

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_admin.py -k fallback_order tests/test_model_utils.py -k persist_fallback`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_admin.py tests/test_model_utils.py api/admin.py config/settings.py
git commit -m "feat: persist fallback order from admin"
```

### Task 2: Lock runtime semantics with tests

**Files:**
- Modify: `tests/test_admin.py`
- Modify: `config/settings.py`

**Step 1: Write the failing test**

Add a test asserting saving fallback order with `persist_default_for_next_restart=True` does not change the current runtime active model returned by admin status when a runtime override exists.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_admin.py -k runtime_override`
Expected: FAIL if the save flow incorrectly overwrites runtime state.

**Step 3: Write minimal implementation**

Ensure the endpoint only persists `.env` values and clears cached settings, without calling `set_active_model(...)` or mutating provider sticky runtime state.

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_admin.py -k runtime_override`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_admin.py api/admin.py config/settings.py
git commit -m "test: preserve runtime override when saving fallback order"
```

### Task 3: Add drag-and-drop admin UI with explicit save/reset

**Files:**
- Modify: `static/admin/index.html`
- Modify: `static/admin/admin.js`
- Modify: `static/admin/styles/legacy.css`

**Step 1: Write the failing test**

If practical, add or extend admin tests for the expected rendered success message contract from the save endpoint. If no frontend harness exists, use backend contract tests only and verify the UI manually after implementation.

**Step 2: Implement minimal UI**

- Render configured fallback models as draggable rows
- Keep list state in memory
- Add checkbox for `Also set first model as MODEL in .env for next restart`
- Add `Save Order` and `Reset`
- Show dirty state and save status messages

**Step 3: Verify manually**

Run the app locally, open the admin dashboard, reorder models, save, refresh, and confirm the order persists.

**Step 4: Commit**

```bash
git add static/admin/index.html static/admin/admin.js static/admin/styles/legacy.css
git commit -m "feat: add draggable fallback ordering in admin"
```

### Task 4: Full verification

**Files:**
- Test: `tests/test_admin.py`
- Test: `tests/test_model_utils.py`

**Step 1: Run focused regression tests**

Run: `uv run pytest -q tests/test_admin.py tests/test_model_utils.py`
Expected: PASS

**Step 2: Run broader related admin/provider tests**

Run: `uv run pytest -q tests/test_nvidia_nim.py tests/test_admin.py tests/test_model_utils.py`
Expected: PASS

**Step 3: Manual dashboard sanity check**

Confirm:
- drag/drop reorders rows
- unsaved reorder can be reset
- save persists fallback order
- checkbox persists `MODEL` for next restart only
- current runtime active model does not change immediately

**Step 4: Commit**

```bash
git add .
git commit -m "test: verify admin fallback drag-drop persistence"
```
