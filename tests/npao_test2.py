import numpy as np
import time
import random

# Parameters
n_rows = 1000
n_cols = 10
j = 5

# 1. Create lol: 1000 x 10
lol = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]

# ---- a. pure python with accumulator ----
t0 = time.time()
s_a = 0.0
for row in lol:
    s_a += row[j]
t1 = time.time()

# ---- b. sum with generator (no explicit column extraction) ----
t2 = time.time()
s_b = sum(row[j] for row in lol)
t3 = time.time()

# ---- c. npao: convert, slice column, np.sum ----
t4 = time.time()
A_obj = np.array(lol, dtype=object)
col_obj = A_obj[:, j]
s_c = np.sum(col_obj)
t5 = time.time()

# ---- d. npa: convert to float64, slice column, np.sum ----
t6 = time.time()
A_num = np.array(lol, dtype=np.float64)
col_num = A_num[:, j]
s_d = np.sum(col_num)
t7 = time.time()

# Results
print(f"a. python loop : {t1 - t0:.6f}")
print(f"b. sum(gen)    : {t3 - t2:.6f}")
print(f"c. npao        : {t5 - t4:.6f}")
print(f"d. float64     : {t7 - t6:.6f}")

# Sanity check (values should match closely)
print("Results equal:",
      abs(s_a - s_b) < 1e-9,
      abs(s_a - s_c) < 1e-9,
      abs(s_a - s_d) < 1e-9)