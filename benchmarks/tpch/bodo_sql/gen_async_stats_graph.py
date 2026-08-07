import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 3:
    print("Needs two args.")
    sys.exit(-1)

df = pd.read_csv(
    sys.argv[1],
    header=None,
    names=["reserved_cur", "reserved_peak", "used_cur", "used_peak"],
)

x = df.index.to_numpy()

plt.figure(figsize=(14, 6))

plt.plot(x, df["reserved_cur"], label="reserved_cur")
plt.plot(x, df["reserved_peak"], label="reserved_peak")
plt.plot(x, df["used_cur"], label="used_cur")
plt.plot(x, df["used_peak"], label="used_peak")

plt.xlabel("Row index")
plt.ylabel("MB")
plt.title("Async stats for Q" + sys.argv[2])
plt.legend()
plt.tight_layout()
plt.savefig(sys.argv[1] + ".png", dpi=300)
