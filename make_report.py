# scripts/make_report.py
import csv, os, textwrap
IN = "results/results.csv"
OUT = "results/report.md"

def main():
    os.makedirs("results", exist_ok=True)
    rows = list(csv.DictReader(open(IN)))
    header = "| B | H | N | D | Torch (ms) | Triton (ms) | Speedup | Max abs err |"
    sep    = "|---|---|---|---|-----------:|------------:|--------:|------------:|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['B']} | {r['H']} | {r['N']} | {r['D']} | {r['torch_ms']} | {r['triton_ms']} | {r['speedup']} | {r['max_abs_err']} |")
    body = "\n".join(lines)
    md = textwrap.dedent(f"""
    # Benchmark Summary

    ![](../assets/latency_vs_seq.png)
    ![](../assets/speedup_vs_seq.png)

    {body}
    """).strip()
    open(OUT,"w").write(md)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
