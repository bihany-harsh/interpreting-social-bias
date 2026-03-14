# Bug Analysis: `ig2_gpt2_analyze_bias.py` — Gap Computation & File I/O

## Critical Bugs (will crash or produce wrong results)

### Bug 1 — Missing `requires_grad_(True)` on scaled neurons
**Lines 331, 367**

`scaled_neurons_gold` and `scaled_neurons_pred` are returned from `scaled_input()` but never set to require gradients. `torch.autograd.grad(..., patched_mlp_activation)` inside `GPT.forward()` will fail because the tensor isn't tracked by autograd.

The original BERT code has this explicitly (line 339 of `1_analyze_mlm_bias.py`):
```python
scaled_weights.requires_grad_(True)
```

**Fix:** Add after line 331:
```python
scaled_neurons_gold.requires_grad_(True)
```
Add after line 367:
```python
scaled_neurons_pred.requires_grad_(True)
```

---

### Bug 2 — `input_ids` (list) passed to `gpt2_generate` instead of `input_ids_t` (tensor)
**Line 297**

```python
pred_log_prob, generated_tokens, full_sequence = gpt2_generate(model, input_ids, gen_len=T_c, ...)
```

`input_ids` is the raw Python list from line 259. `gpt2_generate` expects a tensor. Should be `input_ids_t`.

**Fix:** Change line 297:
```python
pred_log_prob, generated_tokens, full_sequence = gpt2_generate(model, input_ids_t, gen_len=T_c, max_seq_length=args.max_seq_length)
```

---

### Bug 3 — `generated_tokens` undefined when `--get_ig2_pred` used without `--get_pred`
**Lines 310–312**

```python
if args.get_ig2_pred:
    pred_full_sequence = torch.cat([input_ids_t, generated_tokens], dim=1)
    pred_label = generated_tokens.squeeze(0).tolist()
```

`generated_tokens` is only defined inside the `if args.get_pred:` block (line 297). If you run with `--get_ig2_pred` but NOT `--get_pred`, this will raise `NameError: name 'generated_tokens' is not defined`.

**Fix:** Replace lines 308–312 with:
```python
pred_full_sequence = None
pred_label = None
if args.get_ig2_pred:
    with torch.no_grad():
        _, generated_tokens_pred, _ = gpt2_generate(
            model, input_ids_t, gen_len=T_c, max_seq_length=args.max_seq_length
        )
    pred_full_sequence = torch.cat([input_ids_t, generated_tokens_pred], dim=1)
    pred_label = generated_tokens_pred.squeeze(0).tolist()
```

This generates independently of `get_pred`, so both flags work independently.

---

### Bug 4 — `ig2_gold_filtered` overwrites raw `ig2_gold`, corrupting gap computation
**Lines 401–402 vs 422**

```python
# Line 401-402: BEFORE writing to file
if args.get_ig2_gold_filtered:
    res_dict["ig2_gold"] = convert_to_triplet_ig2(res_dict["ig2_gold"])
```

This replaces the dense `ig2_gold` (shape: `n_layers × 4*d_model`) with sparse triplets `[[layer, neuron, value], ...]`. This filtered form gets written to the jsonl file.

Then the gap computation (line 422) reads it back:
```python
demo1_ig2_gold = np.array(demo1_res_dict['ig2_gold'], np.float32)
```

If `ig2_gold` is now triplets, `np.array()` will produce a wrong shape, and the subtraction `demo1_ig2_gold - demo2_ig2_gold` on line 426 will fail or produce garbage (different triplets selected for demo1 vs demo2).

**Note:** The original BERT code has the exact same bug (lines 366–367 vs 461).

**Fix:** Save BOTH the raw and filtered versions. Replace lines 401–402:
```python
if args.get_ig2_gold_filtered:
    res_dict["ig2_gold_filtered"] = convert_to_triplet_ig2(res_dict["ig2_gold"])
    # Keep raw ig2_gold intact for gap computation
```

Then on line 422, the gap computation reads `ig2_gold` (raw, always dense). The filtered version is stored separately if needed downstream.

---

### Bug 5 — `tokens_info['tokens']` key doesn't exist
**Line 429**

```python
gap_tokens_info = {
    "tokens": demo1_tokens_info['tokens'],   # ← KeyError!
    ...
}
```

Your `tokens_info` dict (lines 281–286) has keys: `tokenized_sentence`, `tokenized_completion`, `gold_demo`, `metadata`. There is no `'tokens'` key. (The BERT code's `example2feature()` returned a different `tokens_info` that had a `'tokens'` key.)

**Fix:** Change line 429:
```python
"tokens": demo1_tokens_info['tokenized_sentence'],
```

---

### Bug 6 — Hardcoded `"Black - White"`
**Line 431**

```python
"gold_obj": "Black - White",
```

Should use the CLI arguments.

**Fix:**
```python
"gold_obj": args.demographic1 + " - " + args.demographic2,
```

---

## Medium Bugs (logical issues)

### Bug 7 — `convert_to_triplet_ig2` on gap values doesn't handle negatives
**Line 437**

```python
if args.get_ig2_gold_gap_filtered:
    gap_res_rmb_dict['ig2_gold_gap'] = convert_to_triplet_ig2(gap_res_rmb_dict['ig2_gold_gap'])
```

`ig2_gold_gap` is a difference (`demo1 - demo2`), so values can be negative. `convert_to_triplet_ig2` filters for `value >= max * 0.1`. If max is small or negative, this threshold is meaningless.

**Fix:** For gap filtering, use absolute values:
```python
def convert_to_triplet_ig2_gap(ig2_list):
    """Like convert_to_triplet_ig2 but uses absolute values for thresholding."""
    ig2 = np.array(ig2_list)
    max_abs_ig2 = np.abs(ig2).max()
    ig2_triplet = []
    for i in range(ig2.shape[0]):
        for j in range(ig2.shape[1]):
            if abs(ig2[i][j]) >= max_abs_ig2 * 0.1:
                ig2_triplet.append([i, j, float(ig2[i][j])])
    return ig2_triplet
```

Then on line 437:
```python
gap_res_rmb_dict['ig2_gold_gap'] = convert_to_triplet_ig2_gap(gap_res_rmb_dict['ig2_gold_gap'])
```

---

### Bug 8 — `base` loaded but unused in gap computation
**Lines 423, 425**

```python
demo1_base = np.array(demo1_res_dict['base'], np.float32)
demo2_base = np.array(demo2_res_dict['base'], np.float32)
```

These are loaded but never used. The variable name `filf_rmb_gap` and `gap_res_rmb_dict` suggest "remove base" (`rm-base`), but no base removal is actually performed.

If the intent is to compute `ig2_gold - base` before the gap (as the "rm-base" naming suggests):
```python
demo1_ig2_gold_rmb = demo1_ig2_gold - demo1_base
demo2_ig2_gold_rmb = demo2_ig2_gold - demo2_base
ig2_gold_gap = demo1_ig2_gold_rmb - demo2_ig2_gold_rmb
```

**Note:** The original BERT code has the same dead code. Decide whether you want base-removed gaps or not.

---

## Minor Bugs

### Bug 9 — Stray `-` in output filename
**Line 247**

```python
output_file = os.path.join(args.output_dir, output_prefix + '-' + demo_label + '-' + '.rlt' + '.jsonl')
```

Produces: `Modifier-ethnicity-N-black-.rlt.jsonl` (note the `-.rlt`).

**Fix:**
```python
output_file = os.path.join(args.output_dir, output_prefix + '-' + demo_label + '.rlt.jsonl')
```

And update the corresponding read paths on lines 413–414 to match:
```python
jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic1 + '.rlt.jsonl'), 'r') as fb, \
jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic2 + '.rlt.jsonl'), 'r') as fw, \
```

---

### Bug 10 — Output files don't include `relation` in name (latent)
**Lines 247, 413–414**

Currently your data has only 1 relation per file (`ET-N-0`), so this works. But if data ever has multiple relations, the `jsonlines.open(..., 'w')` inside the relation loop would overwrite the file each iteration. The original BERT code has the same issue.

**Fix (optional, for robustness):** Either:
- Include `relation` in the filename, OR
- Open the file in `'a'` (append) mode and clear it once before the relation loop

---

## Summary

| # | Severity | Line(s) | Bug | From original? |
|---|----------|---------|-----|----------------|
| 1 | 🔴 Crash | 331, 367 | Missing `requires_grad_(True)` | No (was in original) |
| 2 | 🔴 Crash | 297 | `input_ids` list instead of `input_ids_t` tensor | No |
| 3 | 🔴 Crash | 310–312 | `generated_tokens` undefined without `--get_pred` | No |
| 4 | 🔴 Wrong results | 401–402 | Filtered `ig2_gold` overwrites raw, breaks gap | Yes |
| 5 | 🔴 Crash | 429 | `tokens_info['tokens']` key doesn't exist | Yes (different dict) |
| 6 | 🟡 Wrong output | 431 | Hardcoded "Black - White" | Yes |
| 7 | 🟡 Wrong results | 437 | `convert_to_triplet_ig2` on negative gap values | Yes |
| 8 | 🟡 Dead code | 423, 425 | `base` loaded but unused in gap | Yes |
| 9 | 🟠 Cosmetic | 247 | Stray `-` in filename | No |
| 10 | 🟠 Latent | 247, 413 | No `relation` in filename | Yes |
