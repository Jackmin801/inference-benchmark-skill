import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_model_name(model: str) -> str:
    return model.split("/")[-1]


async def make_prompt(base_url: str, model: str, target_tokens: int) -> tuple[str, int]:
    """Generate a prompt calibrated to approximately target_tokens input tokens."""
    words = ["hi"] * target_tokens
    prompt = " ".join(words)
    url = f"{base_url}/v1/chat/completions"

    async with aiohttp.ClientSession() as session:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1}
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200, f"Calibration failed: {resp.status} {await resp.text()}"
            actual = (await resp.json())["usage"]["prompt_tokens"]

        if abs(actual - target_tokens) / target_tokens > 0.05:
            adjusted = int(len(words) * target_tokens / actual)
            words = ["hi"] * max(1, adjusted)
            prompt = " ".join(words)
            payload["messages"] = [{"role": "user", "content": prompt}]
            async with session.post(url, json=payload) as resp:
                actual = (await resp.json())["usage"]["prompt_tokens"]

    return prompt, actual


async def run_bench(
    base_url: str,
    model: str,
    prompt: str,
    actual_input_tokens: int,
    output_len: int,
    concurrency: int,
    warmup_s: float,
    measure_s: float,
) -> dict:
    """Send streaming requests at given concurrency, measure throughput over a time window."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "stream": True,
        "ignore_eos": True,
    }

    stats = {"output_tokens": 0, "started": 0, "completed": 0, "errors": 0}
    measuring = [False]

    async def worker(session):
        while True:
            stats["started"] += 1
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    stats["errors"] += 1
                    await asyncio.sleep(0.1)
                    continue
                while True:
                    line = await resp.content.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith("data: "):
                        continue
                    if text == "data: [DONE]":
                        break
                    chunk = json.loads(text[6:])
                    choices = chunk.get("choices", [{}])
                    if choices[0].get("delta", {}).get("content"):
                        if measuring[0]:
                            stats["output_tokens"] += 1
            stats["completed"] += 1

    connector = aiohttp.TCPConnector(limit=concurrency + 50, force_close=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(concurrency)]

        print(f"  Warming up ({warmup_s:.0f}s)...", end="", flush=True)
        await asyncio.sleep(warmup_s)
        print(f" measuring ({measure_s}s)...", end="", flush=True)

        measuring[0] = True
        t0 = time.monotonic()
        await asyncio.sleep(measure_s)
        elapsed = time.monotonic() - t0
        measuring[0] = False

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    out_tps = stats["output_tokens"] / elapsed
    total_tps = out_tps * (actual_input_tokens + output_len) / output_len

    print(f" {out_tps:.0f} out tok/s, {total_tps:.0f} total tok/s", flush=True)
    if stats["errors"]:
        print(f"  ({stats['errors']} errors)", flush=True)

    return {
        "output_tokens": stats["output_tokens"],
        "elapsed_s": round(elapsed, 2),
        "output_throughput": round(out_tps, 1),
        "total_throughput": round(total_tps, 1),
        "requests_started": stats["started"],
        "requests_completed": stats["completed"],
        "errors": stats["errors"],
    }


def estimate_warmup(concurrency: int, input_len: int, prefill_results: list[dict], margin: float = 1.5, min_s: float = 5.0, max_s: float = 120.0) -> float:
    """Estimate warmup time from prefill results so all slots finish prefill before measurement."""
    # Find prefill tps at matching or nearest concurrency
    by_conc = {r["concurrency"]: r["total_throughput"] for r in prefill_results}
    if concurrency in by_conc:
        prefill_tps = by_conc[concurrency]
    else:
        # Use nearest concurrency
        nearest = min(by_conc.keys(), key=lambda c: abs(c - concurrency))
        prefill_tps = by_conc[nearest]

    if prefill_tps <= 0:
        return max_s

    total_prefill_tokens = concurrency * input_len
    warmup = (total_prefill_tokens / prefill_tps) * margin
    return max(min_s, min(warmup, max_s))


async def run_sweep(base_url, model, kv_cache_tokens, max_model_len, concurrency_levels, warmup, duration, output_len_override=None, ratios=None, prefill_results=None, warmup_margin=1.5):
    """Run a sweep and return list of result dicts.

    If prefill_results is provided, warmup is computed automatically per config
    based on how long prefill takes to fill all concurrency slots.
    """
    if output_len_override is not None:
        sweep_ratios = [None]
    else:
        sweep_ratios = ratios

    results = []
    prompt_cache = {}

    for ratio in sweep_ratios:
        for concurrency in concurrency_levels:
            seq_len = min(kv_cache_tokens // concurrency, max_model_len - 256)

            if output_len_override is not None:
                output_len = output_len_override
                input_len = seq_len - output_len
                ratio_val = input_len / seq_len
            else:
                input_len = int(seq_len * ratio)
                output_len = seq_len - input_len
                ratio_val = ratio

            if input_len < 1 or output_len < 1 or seq_len < 4:
                print(f"\nSkipping concurrency={concurrency}: seq too small")
                continue

            if prefill_results:
                effective_warmup = estimate_warmup(concurrency, input_len, prefill_results, margin=warmup_margin)
            else:
                effective_warmup = warmup

            print(f"\n{'='*60}")
            print(f"  concurrency={concurrency}  input={input_len}  output={output_len}  seq={input_len+output_len}")
            print(f"{'='*60}")

            if input_len not in prompt_cache:
                prompt, actual = await make_prompt(base_url, model, input_len)
                prompt_cache[input_len] = (prompt, actual)
                print(f"  Calibrated prompt: target={input_len}, actual={actual} tokens")

            prompt, actual_input = prompt_cache[input_len]

            data = await run_bench(
                base_url=base_url,
                model=model,
                prompt=prompt,
                actual_input_tokens=actual_input,
                output_len=output_len,
                concurrency=concurrency,
                warmup_s=effective_warmup,
                measure_s=duration,
            )

            results.append({
                "ratio": round(ratio_val, 4),
                "concurrency": concurrency,
                "input_len": input_len,
                "output_len": output_len,
                **data,
            })

    return results


def plot_prefill(results: list[dict], output_path: Path, model: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    subset = sorted(results, key=lambda r: r["concurrency"])
    xs = [r["concurrency"] for r in subset]
    ys = [r["total_throughput"] for r in subset]
    ax.plot(xs, ys, "o-", color=plt.cm.tab10.colors[0], label="prefill tok/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"Prefill Throughput — {model}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decode(results: list[dict], output_path: Path, model: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    ratios = sorted(set(r["ratio"] for r in results))

    for i, ratio in enumerate(ratios):
        subset = sorted([r for r in results if r["ratio"] == ratio], key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in subset]
        out_ys = [r["output_throughput"] for r in subset]
        total_ys = [r["total_throughput"] for r in subset]
        label = f"{int(ratio*100)}% in / {int((1-ratio)*100)}% out"
        ax.plot(xs, out_ys, "o-", color=colors[i], label=f"{label} — output tok/s")
        ax.plot(xs, total_ys, "s--", color=colors[i], alpha=0.6, label=f"{label} — total tok/s")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"Decode Throughput — {model}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    args,
    prefill_results: list[dict],
    decode_results: list[dict],
    prefill_plot: str,
    decode_plot: str,
):
    model_name = safe_model_name(args.model)
    lines = [
        f"# Benchmark: {args.model}",
        "",
        "## Server Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Model | {args.model} |",
        f"| TP | {args.tp} |",
        f"| Max Model Len | {args.max_model_len:,} |",
        f"| KV Cache Tokens | {args.kv_cache_tokens:,} |",
        f"| KV Cache Dtype | {args.kv_cache_dtype} |",
        f"| Prefix Caching | disabled |",
        f"| Chunked Prefill | enabled |",
        f"| Warmup | {args.warmup}s |",
        f"| Measurement | {args.duration}s |",
        f"| Date | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |",
        "",
    ]

    # Prefill section
    lines += [
        "## Prefill Throughput",
        "",
        f"![Prefill]({prefill_plot})",
        "",
        "| Concurrency | Input Tokens | Seq Len | Prefill tok/s |",
        "|-------------|-------------|---------|---------------|",
    ]
    for r in sorted(prefill_results, key=lambda x: x["concurrency"]):
        lines.append(
            f"| {r['concurrency']:,} | {r['input_len']:,} | {r['input_len']+r['output_len']:,} | {r['total_throughput']:,.0f} |"
        )

    # Decode sections
    ratios = sorted(set(r["ratio"] for r in decode_results))
    lines += [
        "",
        "## Decode Throughput",
        "",
        f"![Decode]({decode_plot})",
        "",
    ]
    for ratio in ratios:
        pct_in = int(ratio * 100)
        pct_out = 100 - pct_in
        lines += [
            f"### {pct_in}% input / {pct_out}% output",
            "",
            "| Concurrency | Input | Output | Seq Len | Out tok/s | Total tok/s |",
            "|-------------|-------|--------|---------|-----------|-------------|",
        ]
        subset = sorted([r for r in decode_results if r["ratio"] == ratio], key=lambda x: x["concurrency"])
        for r in subset:
            lines.append(
                f"| {r['concurrency']:,} | {r['input_len']:,} | {r['output_len']:,} "
                f"| {r['input_len']+r['output_len']:,} | {r['output_throughput']:,.0f} | {r['total_throughput']:,.0f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


async def async_main(args):
    ratios = [float(r) for r in args.ratios.split(",")]
    concurrency_levels = [int(c) for c in args.concurrency_levels.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = safe_model_name(args.model)

    # 1. Prefill benchmark
    print("\n" + "=" * 70)
    print("  PREFILL BENCHMARK (output_len=1)")
    print("=" * 70)
    prefill_results = await run_sweep(
        base_url=args.base_url,
        model=args.model,
        kv_cache_tokens=args.kv_cache_tokens,
        max_model_len=args.max_model_len,
        concurrency_levels=concurrency_levels,
        warmup=args.warmup,
        duration=args.duration,
        output_len_override=1,
    )

    # 2. Decode benchmark
    print("\n" + "=" * 70)
    print("  DECODE BENCHMARK")
    if args.auto_warmup and prefill_results:
        print("  (auto-warmup from prefill results)")
    print("=" * 70)
    decode_results = await run_sweep(
        base_url=args.base_url,
        model=args.model,
        kv_cache_tokens=args.kv_cache_tokens,
        max_model_len=args.max_model_len,
        concurrency_levels=concurrency_levels,
        warmup=args.warmup,
        duration=args.duration,
        ratios=ratios,
        prefill_results=prefill_results if args.auto_warmup else None,
        warmup_margin=args.warmup_margin,
    )

    if not prefill_results and not decode_results:
        print("No successful benchmark runs.")
        return

    # 3. Plots
    prefill_plot = f"prefill_{model_name}.png"
    decode_plot = f"decode_{model_name}.png"

    if prefill_results:
        plot_prefill(prefill_results, output_dir / prefill_plot, args.model)
        print(f"\nPrefill plot saved to {output_dir / prefill_plot}")

    if decode_results:
        plot_decode(decode_results, output_dir / decode_plot, args.model)
        print(f"Decode plot saved to {output_dir / decode_plot}")

    # 4. Markdown report
    md_path = output_dir / f"benchmark_{model_name}.md"
    write_markdown(md_path, args, prefill_results, decode_results, prefill_plot, decode_plot)
    print(f"Report saved to {md_path}")

    # 5. Raw JSON
    json_path = output_dir / f"benchmark_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump({"prefill": prefill_results, "decode": decode_results}, f, indent=2)
    print(f"Raw data saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark: prefill + decode sweep")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-tokens", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (for report metadata)")
    parser.add_argument("--kv-cache-dtype", default="auto", help="KV cache dtype (for report metadata)")
    parser.add_argument("--ratios", default="0.25,0.5", help="Comma-separated input fractions for decode")
    parser.add_argument("--concurrency-levels", default="32,64,128,256,512,1024,2048")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--warmup", type=float, default=10.0, help="Warmup seconds (used for prefill; decode default if auto-warmup is off)")
    parser.add_argument("--duration", type=float, default=15.0, help="Measurement duration in seconds")
    parser.add_argument("--auto-warmup", action="store_true", help="Auto-compute decode warmup from prefill results")
    parser.add_argument("--warmup-margin", type=float, default=1.5, help="Margin multiplier for auto-warmup (default 1.5)")
    parser.add_argument("--output-dir", default="./bench-results")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
