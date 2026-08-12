# IT3103 Assignment 2 — Guide: Building a Multimodal ReAct + Reflexion Agent

Your notebook already gives you the scaffolding (model loading, `vlm_generate`, tool-call
parsing, evaluator). The `# TODO` blocks are what you need to fill in. This guide walks through
each one in order, explains *why* it works the way it does, and gives you enough pseudocode to
implement it yourself without just handing you a finished solution — the rubric explicitly grades
your engineering, not whether it matches a reference implementation.

---

## 0. Before you touch code — plan on paper first

Sketch out your 10 tasks and 3+ tools *before* opening Colab. This saves you from redesigning
mid-way.

**Good task design tips:**
- Mix 2–3 image "genres" (e.g. bar charts, receipts, a photo with small text). The starter code
  gives you chart + receipt generators — extend these or add your own (screenshots, photos).
- At least 3–4 of your 10 tasks should be **unsolvable in one glance** — e.g. "what's the total of
  the two smallest items?" (needs read + calculator) or "what's the value of the 3rd bar from the
  left?" where the label is too small to read without cropping. This is what makes ReAct actually
  *beat* the baseline — if all your tasks are one-shot readable, ReAct won't show improvement and
  you'll lose marks on "measurable improvement over baseline."
- Store `(image, question, answer)` — the `answer` is your ground truth for the evaluator.

**Tool design tips (need ≥3):**
- `crop_region(box)` — near-mandatory, since it's what lets the agent "zoom in."
- `read_text(box)` — a second VLM call scoped to a crop, asking it to transcribe just that region.
- `calculator(expression)` — already implemented for you (safe AST-based eval, don't use `eval()`).
- `finish(answer)` — signals the loop to stop; already referenced in the scaffold.
- Optional 4th tool if you want extra robustness marks: `zoom(factor)` (resize a crop up) or
  `count_objects()` for a counting-style task.

---

## Part A — Task dataset & baseline (Cells 9–12)

**Cell 10** already builds charts/receipts. Extend `TASKS` with your own harder examples — mutate
the existing images with different values, or add new image types with PIL. Each entry needs
`{'image': PIL.Image, 'question': str, 'answer': str}`.

**Cell 12 TODO** — the loop structure is already sketched:
```python
baseline_correct = 0
for i, task in enumerate(TASKS):
    pred = baseline_answer(task)
    ok = is_correct(pred, task['answer'])
    baseline_correct += ok
    print(f"[{i}] pred={pred!r} gold={task['answer']!r} {'OK' if ok else 'X'}")
print(f"Baseline accuracy: {baseline_correct}/{len(TASKS)}")
```
This is essentially complete already in the scaffold — just make sure you **save the per-task
predictions in a list** (not just print them), because you'll want to compare baseline vs ReAct
vs Reflexion per-task later, not just the aggregate number.

```python
baseline_results = []  # add this
for i, task in enumerate(TASKS):
    pred = baseline_answer(task)
    ok = is_correct(pred, task['answer'])
    baseline_results.append({'task_id': i, 'pred': pred, 'ok': ok})
```

---

## Part B — Tools (Cell 14)

`crop_region` — literally the one-liner the comment suggests:
```python
def crop_region(box):
    return current_image.crop(tuple(box))
```
`current_image` is a global set at the start of each agent run (Cell 16 already does this) — this
lets the tool "see" whichever image the current task is using.

`read_text(box=None)` — call the VLM again, scoped to the crop (or full image if no box given):
```python
def read_text(box=None):
    img = current_image.crop(tuple(box)) if box else current_image
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Transcribe all visible text/numbers in this image exactly.'},
    ]}]
    return vlm_generate(messages, max_new_tokens=64)
```
This is the "second model call as a tool" pattern — a common ReAct design where a tool is itself
powered by the LLM but scoped to a narrower sub-problem (much more reliable than asking the model
to read tiny text in the full image at once).

You also need a JSON schema per tool (for the `tools=` argument passed into `vlm_generate`).
Follow the OpenAI/Qwen function-calling schema format:
```python
TOOLS = [
    {"type": "function", "function": {
        "name": "crop_region",
        "description": "Crop a region of the current image for closer inspection.",
        "parameters": {"type": "object", "properties": {
            "box": {"type": "array", "items": {"type": "integer"}, "description": "[x1,y1,x2,y2]"}
        }, "required": ["box"]}
    }},
    {"type": "function", "function": {
        "name": "read_text",
        "description": "Transcribe text/numbers in a region of the image.",
        "parameters": {"type": "object", "properties": {
            "box": {"type": "array", "items": {"type": "integer"}}
        }}
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Submit the final answer and stop.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"}
        }, "required": ["answer"]}
    }},
]
```

---

## Part B — The ReAct loop (Cell 16)

The core skeleton (system prompt, `parse_tool_call`, `current_image`) is provided. You need to
finish `react_agent`. The loop structure is:

```python
def react_agent(task, max_steps=6, verbose=True):
    global current_image
    current_image = task['image']
    trace = []
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': [
            {'type': 'image', 'image': task['image']},
            {'type': 'text', 'text': task['question']},
        ]},
    ]

    for step in range(max_steps):
        out = vlm_generate(messages, tools=TOOLS)
        trace.append(('model', out))
        name, args = parse_tool_call(out)

        if name is None:
            # Model didn't call a tool — treat raw text as a fallback answer,
            # or nudge it to use finish(). Handle malformed calls gracefully here.
            messages.append({'role': 'assistant', 'content': out})
            messages.append({'role': 'user', 'content':
                'Please respond with a tool call, e.g. finish(answer=...).'})
            continue

        messages.append({'role': 'assistant', 'content': out})

        if name == 'finish':
            return args.get('answer', ''), trace

        # Execute the tool
        try:
            if name == 'crop_region':
                result_img = crop_region(args['box'])
                current_image = result_img  # optional: let subsequent crops chain
                tool_msg = {'role': 'tool', 'content': [
                    {'type': 'text', 'text': 'Cropped region:'},
                    {'type': 'image', 'image': result_img},
                ]}
            elif name == 'read_text':
                text = read_text(args.get('box'))
                tool_msg = {'role': 'tool', 'content': text}
            elif name == 'calculator':
                val = calculator(args['expression'])
                tool_msg = {'role': 'tool', 'content': str(val)}
            else:
                tool_msg = {'role': 'tool', 'content': f'Unknown tool: {name}'}
        except Exception as e:
            tool_msg = {'role': 'tool', 'content': f'Error: {e}'}

        trace.append(('tool', name, args))
        messages.append(tool_msg)

    # Ran out of steps without finish()
    return None, trace
```

Key things the rubric checks here:
- **Robustness**: malformed `<tool_call>` JSON, unknown tool names, and tool execution errors
  should not crash the loop — they should feed an error observation back so the model can recover.
- **Image tool results ride as images**, not text descriptions (per the assignment's note) — this
  is why `crop_region`'s tool message embeds a `{'type': 'image', ...}` block.
- **Step limit**: if `max_steps` is hit without `finish()`, return a clear "no answer" rather than
  silently returning `None` with no trace — you'll want this for the failure-case writeup.

Run this over all 10 tasks the same way you did the baseline, and store `react_results` with
predictions + traces (you'll need traces for Part D's failure case documentation).

---

## Part C — Reflexion (Cells 18–20)

**`reflect()`** — build a prompt from the failed attempt and ask the model to diagnose itself:
```python
def reflect(task, pred, trace):
    trace_summary = '\n'.join(
        f"{t[0]}: {t[1] if len(t)==2 else (t[1], t[2])}" for t in trace
    )
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': task['image']},
        {'type': 'text', 'text': (
            f"Question: {task['question']}\n"
            f"Your answer was: {pred!r}, which was incorrect.\n"
            f"Your reasoning trace was:\n{trace_summary}\n\n"
            "In 1-2 sentences, give a SPECIFIC reflection: what went wrong "
            "(e.g. misread a number, cropped the wrong region, arithmetic error), "
            "and what you should do differently next time."
        )},
    ]}]
    return vlm_generate(messages, max_new_tokens=100)
```
Passing the image back in (as the comment notes) matters — a text-only reflection can't diagnose
"I misread the label" type errors since it has no way to re-examine the image.

**`reflexion_agent()`** — wraps `react_agent` in a retry loop, injecting memory each time:
```python
def reflexion_agent(task, max_attempts=3, verbose=True):
    memory = []
    for attempt in range(max_attempts):
        # Inject prior reflections into the system prompt for this attempt
        extra = ''
        if memory:
            extra = '\n\nLessons from previous attempts:\n' + '\n'.join(
                f"- {m}" for m in memory)
        task_with_memory = dict(task)  # or pass `extra` into react_agent's SYSTEM_PROMPT
        pred, trace = react_agent_with_context(task, extra_system=extra)

        if evaluate(pred, task):
            return pred, attempt + 1, memory
        if attempt < max_attempts - 1:
            reflection = reflect(task, pred, trace)
            memory.append(reflection)

    return pred, max_attempts, memory  # final (failed) attempt
```
You'll need a small variant of `react_agent` that accepts an extra system-prompt string (or just
prepend `extra` as an additional user message before the question) so reflections actually reach
the model on the next attempt.

**Comparison table (Cell 20)** — once you have `baseline_results`, `react_results`, and
`reflexion_results`, this is just aggregation:
```python
import pandas as pd

def acc(results): return sum(r['ok'] for r in results) / len(results)

print(f"Baseline   {sum(r['ok'] for r in baseline_results)}/{len(TASKS)}")
print(f"ReAct      {sum(r['ok'] for r in react_results)}/{len(TASKS)}")
print(f"Reflexion  {sum(r['ok'] for r in reflexion_results)}/{len(TASKS)}")

# Attempt-level breakdown for Reflexion
from collections import Counter
attempt_counts = Counter(r['attempts_used'] for r in reflexion_results if r['ok'])
print("Solved on attempt:", dict(attempt_counts))
```
Screenshot this table (or render it as a small dataframe/markdown) for your report.

---

## Part D — Report (max 4 pages, separate `report.pdf`)

Structure it directly against the rubric so nothing gets missed:

1. **Setup** (short) — which model (8B/4B), quantisation config, `max_pixels`/`min_pixels`, any
   OOM handling you did.
2. **Task dataset** — briefly list/describe your 10 tasks (a table with question + answer is
   fine), and say which ones require multi-step tool use vs which are "easy."
3. **Tool design justification** — for each tool: what it does and *why you chose it* relative to
   your tasks (e.g. "read_text exists because bar-chart labels are unreadable below ~15px at the
   capped resolution").
4. **≥3 documented failure cases** — for each: the question, the model's wrong reasoning trace
   (a few key steps, not the full raw dump), and whether ReAct or Reflexion subsequently fixed it
   (and why/why not).
5. **Comparison table** — Baseline vs ReAct vs Reflexion accuracy, plus the attempt breakdown.
6. **Critical reflection on small-model agents** — this is where the marks for "insightful
   discussion" live. Concretely address: did you see hallucinated tool calls, JSON parsing
   failures, counting errors, arithmetic slips? Does ReAct actually help, or does the model
   sometimes call tools unnecessarily/incorrectly? Tie this back to the assignment's framing
   question — can an 8B/4B model reason and act effectively, and under what conditions does it
   fail?

---

## Submission checklist

- [ ] `agent_qwen.ipynb` — run top to bottom in Colab, **do not clear outputs** before saving.
- [ ] `report.pdf` — ≤ 4 pages.
- [ ] `README.md` — one paragraph: model used, how to run, known issues.
- [ ] `data/` folder with your task images.
- [ ] Zipped as `<studentID>_IT3103_ASSN2.zip`.
- [ ] Don't hard-code answers to your tasks anywhere in the agent logic — the rubric explicitly
      calls this out, and it's also flatly against the point of the assignment.

---

## A few practical Colab gotchas

- If the 8B model OOMs on the T4, flip `USE_FALLBACK_4B = True` and re-run the load cell — this is
  explicitly accepted, just state it in your report.
- If generation is slow, lower `MAX_PIXELS` — but check your `read_text` tasks still work at the
  new resolution (small text is what suffers first).
- Set `do_sample=False` (already default in `vlm_generate`) so your accuracy numbers are
  reproducible run to run — useful since you're comparing three conditions.
