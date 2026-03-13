# Timeout-Aware Model Fallback Design

**Date:** 2026-03-13

## Summary

Extend NVIDIA NIM model handling so the app accepts the newly requested model IDs `nvidia/nemotron-3-super-120b-a12b` and `qwen/qwen3.5-122b-a10b`, including user-entered leading-slash aliases like `/nvidia/nemotron-3-super-120b-a12b`.

Improve upstream timeout handling so timeout-heavy models fail over more cleanly and the admin UI explains what happened instead of only showing a raw `Request timed out.` string.

## Current State

- `config/settings.py` defines the default fallback order and quick-switch model list.
- `providers/model_utils.py` resolves model aliases only by exact catalog match or unique tail match.
- `providers/nvidia_nim/client.py` retries across keys and models, but timeout exceptions are treated as generic failures.
- `api/admin.py` aggregates per-model recent errors from raw request log strings.
- `static/admin/admin.js` renders `Last error: ...` using the raw text without context.
- `nvidia_nim_models.json` is stale and does not include the newly requested models.

## Goals

- Support the two requested NVIDIA models in local defaults.
- Normalize leading-slash model IDs to canonical IDs.
- Detect timeout errors explicitly and treat them as fallback-worthy transport failures.
- Record timeout failures against the per-model circuit breaker so repeatedly slow models are deprioritized.
- Improve admin diagnostics so timeout-caused fallbacks are understandable at a glance.

## Non-Goals

- Dynamic model catalog refresh from the network.
- Adaptive per-model timeout budgets.
- Major admin dashboard redesign.

## Chosen Approach

### 1. Normalize requested model IDs

Update model normalization to:

- strip a single leading slash before catalog lookup
- return canonical model IDs when the slashless value matches the catalog
- preserve existing tail-match behavior for shorthand names

This keeps user input flexible without changing the stored canonical IDs.

### 2. Refresh local model defaults

Update the local default fallback chain to include:

- `nvidia/nemotron-3-super-120b-a12b`
- `qwen/qwen3.5-122b-a10b`

Also refresh `nvidia_nim_models.json` entries minimally so alias resolution can recognize these IDs locally.

### 3. Make timeout handling explicit

In the NVIDIA provider:

- classify `openai.APITimeoutError`, `openai.APIConnectionError`, and `httpx.TimeoutException` as timeout/network failures
- count timeout failures as model failures for circuit-breaker purposes
- keep bad-request failures fail-fast
- continue trying other keys for the current model, then fall back to the next model if all attempts for that model time out

This preserves the current retry structure while making timeout-heavy models naturally lose stickiness and eventually get skipped.

### 4. Improve operator-facing error text

Summarize recent model errors in the admin API so timeout failures become clearer, for example:

- `Timed out after 120s`
- `Timed out after 120s, switched to qwen/qwen3.5-122b-a10b`

The UI will still render one short line, but the text will be curated instead of blindly echoing the raw exception.

## Testing Strategy

- Add failing tests for leading-slash alias normalization.
- Add failing tests for timeout-triggered model fallback in non-streaming completions.
- Add failing tests for stream timeout fallback behavior.
- Add failing tests for timeout error mapping and admin error summarization if needed.

## Risks

- Over-classifying all connection errors as timeout-like could open model circuit breakers too aggressively.
- Refreshing the local catalog manually can drift again later.

## Mitigations

- Scope timeout classification to known timeout/connection exception classes only.
- Keep catalog changes limited to the requested models and document that the file is a local snapshot.
