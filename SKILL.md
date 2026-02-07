---
name: bench-sweep
description: Run a vLLM throughput benchmark sweep across concurrency levels and input/output ratios, then generate a throughput graph.
disable-model-invocation: true
---

# Benchmark Throughput Sweep

Time-based benchmark: sends streaming requests at max concurrency, measures throughput over a fixed window, then cancels and moves on. Each config point takes ~25 seconds (10s warmup + 15s measurement) instead of waiting for all requests to complete.

Collect these parameters from the user (defaults shown):

- `--base-url` (default: `http://localhost:8000`)
- `--model` (required — the model name served by vLLM)
- `--kv-cache-tokens` (required — total KV cache capacity in tokens, from server logs)
- `--ratios` (default: `0.25,0.5` — input token fraction of total sequence length)
- `--concurrency-levels` (default: `32,64,128,256,512,1024,2048`)
- `--max-model-len` (default: `65536`)
- `--warmup` (default: `10.0` — seconds to warm up before measuring)
- `--duration` (default: `15.0` — seconds to measure throughput)
- `--output-dir` (default: `./bench-results`)

Then run the script:

```bash
uv run python .claude/skills/bench-sweep/scripts/bench_sweep.py \
  --base-url <url> \
  --model <model> \
  --kv-cache-tokens <tokens> \
  --ratios <ratios> \
  --concurrency-levels <levels> \
  --warmup <seconds> \
  --duration <seconds> \
  --output-dir <dir>
```

After the script completes, show the user the generated throughput graph at `<output-dir>/throughput_sweep.png`.
