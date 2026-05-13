import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("attention_benchmarks.csv")

# ----------------------------------------
# runtime plot
# ----------------------------------------
plt.figure(figsize=(10, 6))

for kernel in df["kernel"].unique():

    sub = df[df["kernel"] == kernel]

    plt.plot(
        sub["N"],
        sub["time_ms"],
        marker="o",
        label=kernel
    )

plt.xscale("log", base=2)

plt.xlabel("Sequence Length (N)")
plt.ylabel("Runtime (ms)")
plt.title("Attention Kernel Benchmark")

plt.legend()
plt.grid(True)

plt.savefig("attention_runtime.png", dpi=300)

plt.show()

# speedup vs naive kernels only

naive = df[df["kernel"] == "naive_attention"]

my_kernels = [
    "tiled_attention",
    "tiled_attention_2",
    "online_softmax",
    "online_warp",
]

speedup_rows = []

for kernel in my_kernels:

    sub = df[df["kernel"] == kernel]

    merged = sub.merge(
        naive,
        on="N",
        suffixes=("", "_naive")
    )

    merged["speedup"] = (
        merged["time_ms_naive"] /
        merged["time_ms"]
    )

    merged["kernel"] = kernel

    speedup_rows.append(
        merged[["kernel", "N", "speedup"]]
    )

speedup_df = pd.concat(speedup_rows)

plt.figure(figsize=(10, 6))

for kernel in my_kernels:

    sub = speedup_df[
        speedup_df["kernel"] == kernel
    ]

    plt.plot(
        sub["N"],
        sub["speedup"],
        marker="o",
        label=kernel
    )

plt.xscale("log", base=2)

plt.xlabel("Sequence Length (N)")
plt.ylabel("Speedup vs Naive")
plt.title("Custom CUDA Attention Kernel Speedup")

plt.legend()
plt.grid(True)

plt.savefig("attention_speedup.png", dpi=300)

plt.show()