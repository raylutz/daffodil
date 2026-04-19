import numpy as np
import time

n = 1_000_000

lf = [float(i) for i in range(n)]
npao = np.array(lf, dtype=object)
npa = np.array(lf, dtype=np.float64)

# 1. Python loop
t0 = time.time()
s0 = 0.0
for x in lf:
    s0 += x
t1 = time.time()

# 2. Python sum
t2 = time.time()
s1 = sum(lf)
t3 = time.time()

# 3. np.sum on object array
t4 = time.time()
s2 = np.sum(npao)
t5 = time.time()

# 4. np.sum on float64
t6 = time.time()
s3 = np.sum(npa)
t7 = time.time()

print("python loop:", t1 - t0)
print("python sum :", t3 - t2)
print("npao sum   :", t5 - t4)
print("float64 sum:", t7 - t6)
