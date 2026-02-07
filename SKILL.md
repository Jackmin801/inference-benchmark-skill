---
name: vllm-benchmark
description: Run prefill + decode throughput benchmarks against a vLLM server and generate a markdown report with plots.
disable-model-invocation: true
---

# vLLM Benchmark

Runs both prefill and decode throughput sweeps in a single invocation. Time-based measurement: sends streaming requests at max concurrency, measures throughput over a fixed window, then cancels and moves on.

Outputs per model:
- `benchmark_{model}.md` — markdown report with server config, prefill table, decode tables
- `benchmark_{model}.json` — raw data
- `prefill_{model}.png` — prefill throughput plot
- `decode_{model}.png` — decode throughput plot

Collect these parameters from the user (defaults shown):

- `--base-url` (default: `http://localhost:8000`)
- `--model` (required — the model name served by vLLM)
- `--kv-cache-tokens` (required — total KV cache capacity in tokens, from server logs)
- `--tp` (default: `1` — tensor parallel size, for report metadata)
- `--kv-cache-dtype` (default: `auto` — for report metadata)
- `--ratios` (default: `0.25,0.5` — input token fractions for decode sweep)
- `--concurrency-levels` (default: `32,64,128,256,512,1024,2048`)
- `--max-model-len` (default: `65536`)
- `--warmup` (default: `10.0` — seconds to warm up before measuring)
- `--duration` (default: `15.0` — seconds to measure throughput)
- `--output-dir` (default: `./bench-results`)

Then run:

```bash
uv run python .claude/skills/vllm-benchmark-skill/scripts/bench_sweep.py \
  --base-url <url> \
  --model <model> \
  --kv-cache-tokens <tokens> \
  --tp <tp_size> \
  --output-dir <dir>
```

After the script completes, show the user the generated markdown report at `<output-dir>/benchmark_{model}.md`.
