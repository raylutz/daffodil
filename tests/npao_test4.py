import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------
# 1. create 1000x1000 lolf
# -------------------------
n_rows = 1000
n_cols = 1000

big_lol = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]

# -------------------------
# 2. column selection steps
# -------------------------
col_steps = list(range(10, n_cols + 1, 10))

# Results
t_gen = []
t_loop = []
t_npao = []
t_npa = []
t_pd = []

# -------------------------
# Benchmark
# -------------------------
for k in col_steps:

    # build exact-size data OUTSIDE timer
    lol = [row[:k] for row in big_lol]

    cols = list(range(0, k))   # first k columns

    # ---- a. Python generator (sum of sums)
    t0 = time.time()
    s = 0.0
    for j in cols:
        s += sum(row[j] for row in lol)
    t1 = time.time()
    t_gen.append(t1 - t0)

    # ---- b. Python nested loop
    t2 = time.time()
    s = 0.0
    for row in lol:
        for j in cols:
            s += row[j]
    t3 = time.time()
    t_loop.append(t3 - t2)

    # ---- c. npao
    t4 = time.time()
    A_obj = np.array(lol, dtype=object)
    s = 0.0
    for j in cols:
        s += np.sum(A_obj[:, j])
    t5 = time.time()
    t_npao.append(t5 - t4)

    # ---- d. float64
    t6 = time.time()
    A_num = np.array(lol, dtype=np.float64)
    s = 0.0
    for j in cols:
        s += np.sum(A_num[:, j])
    t7 = time.time()
    t_npa.append(t7 - t6)

    # ---- e. pandas (conversion + compute)
    t8 = time.time()
    df = pd.DataFrame(lol)
    col_sums = df.iloc[:, :k].sum()   # ensures pure numeric reduction
    t9 = time.time()
    t_pd.append(t9 - t8)

# -------------------------
# 4. Plot
# -------------------------
plt.figure(figsize=(10, 6))

plt.plot(col_steps, t_gen, label="python sum(gen)")
plt.plot(col_steps, t_loop, label="python nested loop")
plt.plot(col_steps, t_npao, label="npao")
plt.plot(col_steps, t_npa, label="float64")
plt.plot(col_steps, t_pd, label="pandas")

plt.xlabel("Number of columns")
plt.ylabel("Time (seconds)")
plt.title("Column Sum Performance vs Number of Columns")
plt.legend()
plt.grid(True)

plt.show()
