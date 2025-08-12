# scripts/plot_results.py
import os, csv
import matplotlib.pyplot as plt

IN  = "results/results.csv"
OUT_LAT = "assets/latency_vs_seq.png"
OUT_SPD = "assets/speedup_vs_seq.png"

def load():
    Ns, torch_ms, triton_ms, speedup = [], [], [], []
    with open(IN) as f:
        r = csv.DictReader(f)
        for row in r:
            Ns.append(int(row["N"]))
            torch_ms.append(float(row["torch_ms"]))
            triton_ms.append(float(row["triton_ms"]))
            speedup.append(float(row["speedup"]))
    return Ns, torch_ms, triton_ms, speedup

def main():
    os.makedirs("assets", exist_ok=True)
    Ns, torch_ms, triton_ms, speedup = load()

    # Latency vs N
    plt.figure()
    plt.plot(Ns, torch_ms, marker="o", label="PyTorch SDPA (MATH)")
    plt.plot(Ns, triton_ms, marker="o", label="Triton fused SDPA")
    plt.xlabel("Sequence length N"); plt.ylabel("Latency (ms)")
    plt.title("Attention latency vs sequence length")
    plt.legend(); plt.grid(True, alpha=.2); plt.tight_layout()
    plt.savefig(OUT_LAT, dpi=160)

    # Speedup vs N
    plt.figure()
    plt.plot(Ns, speedup, marker="o")
    plt.xlabel("Sequence length N"); plt.ylabel("Speedup (×)")
    plt.title("Triton vs PyTorch MATH speedup")
    plt.grid(True, alpha=.2); plt.tight_layout()
    plt.savefig(OUT_SPD, dpi=160)

    print(f"Saved {OUT_LAT} and {OUT_SPD}")

if __name__ == "__main__":
    main()
