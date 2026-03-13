# Timeout-Aware Model Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the requested NVIDIA model IDs, normalize leading-slash aliases, and handle timeout-driven fallback more clearly in the provider and admin UI.

**Architecture:** Keep the change local to the existing NVIDIA NIM provider flow. First lock the expected timeout and alias behaviors with focused tests, then make the smallest implementation changes in model normalization, provider error handling, and admin summarization to satisfy those tests.

**Tech Stack:** Python 3.10+, FastAPI, pytest, OpenAI Python SDK, httpx, vanilla JS admin UI

---

### Task 1: Lock model alias expectations with tests

**Files:**
- Modify: `tests/test_nvidia_nim.py`
- Modify: `providers/model_utils.py`
- Modify: `nvidia_nim_models.json`

**Step 1: Write the failing test**

Add tests asserting:

```python
assert resolve_model_alias("/nvidia/nemotron-3-super-120b-a12b") == "nvidia/nemotron-3-super-120b-a12b"
assert resolve_model_alias("/qwen/qwen3.5-122b-a10b") == "qwen/qwen3.5-122b-a10b"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k model_alias`
Expected: FAIL because leading-slash aliases are not normalized and the catalog lacks these IDs.

**Step 3: Write minimal implementation**

- Add the two requested model IDs to `nvidia_nim_models.json`
- Update `providers/model_utils.py` to strip one leading slash before exact or tail catalog matching

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k model_alias`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_nvidia_nim.py providers/model_utils.py nvidia_nim_models.json
git commit -m "feat: normalize requested nim model aliases"
```

### Task 2: Lock timeout fallback behavior with tests

**Files:**
- Modify: `tests/test_nvidia_nim.py`
- Modify: `providers/nvidia_nim/client.py`
- Modify: `providers/nvidia_nim/errors.py`

**Step 1: Write the failing test**

Add tests asserting:

```python
with patch.object(nim_provider, "_candidate_models", return_value=["model-a", "model-b"]):
    # first model times out, second succeeds
    result = await nim_provider.complete(req)
    assert result["id"] == "ok"
```

and for streaming:

```python
events = [event async for event in nim_provider.stream_response(req)]
assert any("message_stop" in event for event in events)
```

with the first model raising `openai.APITimeoutError` and the second succeeding.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k timeout`
Expected: FAIL because timeout handling is generic and not explicitly covered by tests.

**Step 3: Write minimal implementation**

- Add timeout/network helper classification methods in `providers/nvidia_nim/client.py`
- Count timeout/network failures as circuit-breaker failures
- Keep falling through to the next model after current-model attempts are exhausted
- Improve `providers/nvidia_nim/errors.py` mapping so timeout/connection errors produce clearer messages

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k timeout`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_nvidia_nim.py providers/nvidia_nim/client.py providers/nvidia_nim/errors.py
git commit -m "fix: handle nim timeout fallback explicitly"
```

### Task 3: Update defaults and admin diagnostics

**Files:**
- Modify: `config/settings.py`
- Modify: `api/admin.py`
- Modify: `static/admin/admin.js`

**Step 1: Write the failing test**

Add or extend tests asserting:

```python
assert "nvidia/nemotron-3-super-120b-a12b" in DEFAULT_NIM_MODEL_FALLBACK_ORDER
assert "qwen/qwen3.5-122b-a10b" in DEFAULT_NIM_MODEL_FALLBACK_ORDER
```

and, if practical, unit-test a helper that summarizes timeout errors for admin display.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k fallback_order`
Expected: FAIL because the new models are not in the default order yet.

**Step 3: Write minimal implementation**

- Insert the new models into `DEFAULT_NIM_MODEL_FALLBACK_ORDER`
- Add a small helper in `api/admin.py` to shorten timeout-heavy raw errors into clearer text
- Keep the admin JS rendering simple and continue using the summarized string

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_nvidia_nim.py -k fallback_order`
Expected: PASS

**Step 5: Commit**

```bash
git add config/settings.py api/admin.py static/admin/admin.js tests/test_nvidia_nim.py
git commit -m "feat: expose clearer timeout fallback diagnostics"
```

### Task 4: Full verification

**Files:**
- Test: `tests/test_nvidia_nim.py`

**Step 1: Run focused regression tests**

Run: `uv run pytest -q tests/test_nvidia_nim.py`
Expected: PASS

**Step 2: Run related tests if they exist**

Run: `uv run pytest -q tests/test_response_conversion.py tests/test_streaming_errors.py`
Expected: PASS

**Step 3: Sanity-check admin UI strings**

Run the app locally and verify the model performance cards render the improved `Last error` text without layout breakage.

**Step 4: Commit**

```bash
git add .
git commit -m "test: verify timeout-aware nim fallback changes"
```
