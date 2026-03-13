# Admin Fallback Order Drag-and-Drop Design

**Date:** 2026-03-13

## Summary

Make the admin dashboard fallback chain reorderable by drag and drop. Saving the new order must persist `NVIDIA_NIM_FALLBACK_MODELS` to `.env` so the order survives restart. An optional checkbox should also persist `MODEL` in `.env` to the first reordered model for the next restart only, without changing the current runtime active model.

## Current State

- The fallback chain is rendered as a static list in `static/admin/admin.js`.
- The backend exposes `POST /admin/fallback-order` in `api/admin.py`, but it is a placeholder that returns `not_implemented`.
- `config/settings.py` already persists `MODEL` to `.env` through `persist_model_to_env`.
- Effective request ordering already uses the cached configured fallback list on each request via `get_model_fallback_chain(...)`.

## Goals

- Drag/drop reorder the configured fallback model list in the admin dashboard.
- Save the reordered list explicitly instead of auto-saving on every drag.
- Persist `NVIDIA_NIM_FALLBACK_MODELS` to `.env`.
- Optionally persist `MODEL` to `.env` as the first reordered model for the next restart only.
- Keep the running process fast by avoiding new request-path file I/O or parsing.
- Apply the saved fallback order to new requests immediately after save by refreshing cached settings.

## Non-Goals

- Changing the current runtime active model when saving fallback order.
- Building a separate modal workflow for model ordering.
- Adding background polling or auto-save behavior.

## Chosen UX

### Dashboard fallback card

- Replace the static fallback rows with draggable rows.
- Each row shows a drag handle, model name, and ordinal label.
- Reordering marks the card as dirty.
- The card shows:
  - `Save Order`
  - `Reset`
  - checkbox: `Also set first model as MODEL in .env for next restart`
- Success feedback distinguishes between:
  - fallback order saved
  - fallback order saved plus default `MODEL` persisted for next restart

### Runtime semantics

- The draggable list edits the configured fallback models only.
- The current runtime override stays untouched.
- Effective request order remains:
  - current active model first
  - configured fallback list after it, with duplicates removed

## Backend Design

### Persistence helpers

Add a shared `.env` writer in `config/settings.py` so both `MODEL` and `NVIDIA_NIM_FALLBACK_MODELS` updates use the same path.

Add a new helper to:

- normalize each model ID through existing alias resolution
- reject an empty list
- dedupe while preserving order
- write `NVIDIA_NIM_FALLBACK_MODELS`
- optionally write `MODEL` to the first model in the new order
- clear cached settings after writing

### Admin API

Replace the placeholder `POST /admin/fallback-order` with a real endpoint that accepts:

- `models: list[str]`
- `persist_default_for_next_restart: bool`

The endpoint returns the updated configured fallback models and current runtime model metadata so the dashboard can refresh without reloading the page.

## Performance Constraints

- No new work in the hot request path.
- No per-request `.env` reads.
- No provider-side file writes or extra parsing during completion/streaming.
- One cache refresh per explicit admin save is acceptable and bounded.

## Testing Strategy

- unit tests for `.env` persistence helpers
- admin endpoint tests for:
  - fallback order persistence
  - optional `MODEL` persistence
  - preserving current runtime active model
- frontend-oriented admin tests where feasible for dirty-state and status behavior
- focused regression tests to ensure effective model chain still prepends runtime override

## Risks

- Accidental save after drag could rewrite `.env` incorrectly
- Drag/drop can be brittle if implemented with complex DOM state

## Mitigations

- Explicit `Save Order` and `Reset`
- Keep drag/drop implementation minimal with native HTML drag events
- Re-render from a single JS state array after each move
