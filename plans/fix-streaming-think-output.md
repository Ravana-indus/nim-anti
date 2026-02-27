# Plan: Fix Streaming Think Output

## Problem
The user reports "system streaming think" appearing in output. This likely means:
1. `<thinking>` tags are appearing in the final output, OR
2. Thinking content is being displayed incorrectly

## Analysis

### Current Flow
1. **NVIDIA NIM Provider** (`providers/nvidia_nim/client.py`):
   - Receives streaming response from API
   - Extracts thinking via TWO paths:
     - `reasoning_content` attribute (lines 324-328)
     - `delta.content` with `<think>` tags via ThinkTagParser (lines 330-335)
   - Emits SSE events

2. **ThinkTagParser** (`providers/nvidia_nim/utils/think_parser.py`):
   - Correctly strips `<think>` and `</think>` tags
   - Returns content WITHOUT tags (tested)

3. **Event Parser** (`messaging/event_parser.py`):
   - Lines 86-87: Parses `thinking_delta` events
   - Returns thinking content (without tags)

4. **Handler** (`messaging/handler.py`):
   - Lines 603-612: Displays thinking in code block

### Potential Issues

1. **Duplicate Thinking**: If a model sends BOTH `reasoning_content` AND content with `<think>` tags, both paths emit thinking → duplicate output

2. **Edge Case**: Some models might send thinking in unexpected formats

3. **Tag Not Stripped**: The ThinkTagParser might not handle all edge cases

## Fix Plan

### Fix 1: Prevent Duplicate Thinking
In `providers/nvidia_nim/client.py`, around line 330, add a check to skip ThinkTagParser processing if `reasoning_content` was already processed:

```python
# After line 328, add:
if reasoning:
    # Skip ThinkTagParser if we already handled reasoning_content
    # to avoid duplicate thinking output
    continue
```

### Fix 2: Ensure Complete Tag Stripping
Verify ThinkTagParser handles all edge cases in `think_parser.py`.

### Fix 3: Add Debug Logging
Add logging to track when thinking is being emitted to help diagnose issues.

## Implementation Steps

1. [ ] Modify `providers/nvidia_nim/client.py` to prevent duplicate thinking
2. [ ] Add debug logging for thinking emission
3. [ ] Test with models that send thinking (e.g., minimax-m2.5, kimi-k2-thinking)
4. [ ] Verify thinking tags don't appear in output
