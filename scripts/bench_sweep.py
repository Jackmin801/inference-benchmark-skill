import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


async def make_prompt(base_url: str, model: str, target_tokens: int) -> tuple[str, int]:
    """Generate a prompt calibrated to approximately target_tokens input tokens."""
    # Start conservatively: each "hi " is ~1-2 tokens for most tokenizers
    words = ["hi"] * target_tokens
    prompt = " ".join(words)
    url = f"{base_url}/v1/chat/completions"

    async with aiohttp.ClientSession() as session:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1}
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200, f"Calibration failed: {resp.status} {await resp.text()}"
            actual = (await resp.json())["usage"]["prompt_tokens"]

        # Adjust proportionally if off by more than 5%
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

        print(f"  Warming up ({warmup_s}s)...", end="", flush=True)
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
    # Estimate total throughput assuming steady-state amortized prefill
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


def plot_results(results: list[dict], output_dir: Path, fixed_output_len: int | None = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors

    if fixed_output_len is not None:
        # Single series: all points have the same fixed output len
        subset = sorted(results, key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in subset]
        total_ys = [r["total_throughput"] for r in subset]
        ax.plot(xs, total_ys, "o-", color=colors[0], label=f"prefill tok/s (output_len={fixed_output_len})")
    else:
        ratios = sorted(set(r["ratio"] for r in results))
        for i, ratio in enumerate(ratios):
            subset = sorted([r for r in results if r["ratio"] == ratio], key=lambda r: r["concurrency"])
            xs = [r["concurrency"] for r in subset]
            out_ys = [r["output_throughput"] for r in subset]
            total_ys = [r["total_throughput"] for r in subset]

            label = f"{int(ratio*100)}% in / {int((1-ratio)*100)}% out"
            ax.plot(xs, out_ys, "o-", color=colors[i], label=f"{label} - output tok/s")
            ax.plot(xs, total_ys, "s--", color=colors[i], alpha=0.6, label=f"{label} - total tok/s")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("vLLM Throughput Sweep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "throughput_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGraph saved to {out_path}")


def print_summary(results: list[dict]):
    print(f"\n{'='*90}")
    print(f"{'Ratio':>6} {'Conc':>6} {'SeqLen':>8} {'In':>6} {'Out':>6} {'Out tok/s':>12} {'Total tok/s':>12} {'Errors':>8}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: (x["ratio"], x["concurrency"])):
        print(
            f"{r['ratio']:>6.2f} {r['concurrency']:>6} {r['input_len']+r['output_len']:>8} "
            f"{r['input_len']:>6} {r['output_len']:>6} {r['output_throughput']:>12.1f} {r['total_throughput']:>12.1f} {r['errors']:>8}"
        )
    print(f"{'='*90}")


async def async_main(args):
    if args.fixed_output_len is not None:
        ratios = [None]
    else:
        ratios = [float(r) for r in args.ratios.split(",")]
    concurrency_levels = [int(c) for c in args.concurrency_levels.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    prompt_cache = {}

    for ratio in ratios:
        for concurrency in concurrency_levels:
            seq_len = min(args.kv_cache_tokens // concurrency, args.max_model_len - 256)

            if args.fixed_output_len is not None:
                output_len = args.fixed_output_len
                input_len = seq_len - output_len
                ratio_val = input_len / seq_len
            else:
                input_len = int(seq_len * ratio)
                output_len = seq_len - input_len
                ratio_val = ratio

            if input_len < 1 or output_len < 1 or seq_len < 4:
                print(f"\nSkipping concurrency={concurrency}: seq_len={seq_len} too small")
                continue

            print(f"\n{'='*60}")
            print(f"  concurrency={concurrency}  input={input_len}  output={output_len}  seq={input_len+output_len}")
            print(f"{'='*60}")

            if input_len not in prompt_cache:
                prompt, actual = await make_prompt(args.base_url, args.model, input_len)
                prompt_cache[input_len] = (prompt, actual)
                print(f"  Calibrated prompt: target={input_len}, actual={actual} tokens")

            prompt, actual_input = prompt_cache[input_len]

            data = await run_bench(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                actual_input_tokens=actual_input,
                output_len=output_len,
                concurrency=concurrency,
                warmup_s=args.warmup,
                measure_s=args.duration,
            )

            results.append({
                "ratio": ratio_val,
                "concurrency": concurrency,
                "input_len": input_len,
                "output_len": output_len,
                **data,
            })

    if not results:
        print("No successful benchmark runs.")
        return

    print_summary(results)
    plot_results(results, output_dir, fixed_output_len=args.fixed_output_len)

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'sweep_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark sweep (time-based)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-tokens", type=int, required=True)
    parser.add_argument("--ratios", default="0.25,0.5", help="Comma-separated input fractions of seq length")
    parser.add_argument("--fixed-output-len", type=int, default=None, help="Override output_len to a fixed value (ignores ratios)")
    parser.add_argument("--concurrency-levels", default="32,64,128,256,512,1024,2048")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--warmup", type=float, default=10.0, help="Warmup seconds before measuring")
    parser.add_argument("--duration", type=float, default=15.0, help="Measurement duration in seconds")
    parser.add_argument("--output-dir", default="./bench-results")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
