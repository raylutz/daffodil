import timeit

def function1(gkeys, keydict, silent_error):
    idxs = []
    for gkey in gkeys:
        idx = keydict.get(gkey, -1)
        if idx >= 0:
            idxs.append(idx)
        elif not silent_error:
            raise KeyError

def function2(gkeys, keydict, silent_error):
    idxs = []
    for gkey in gkeys:
        if gkey in keydict:
            idxs.append(keydict[gkey])
        elif not silent_error:
            raise KeyError

# Define test data
gkeys = [str(i) for i in range (1000)]
keydict = {str(k): idx for idx, k in enumerate(range(10000))}
silent_error = True  # Set to True to suppress KeyError

# Measure execution time for function 1
time_function1 = timeit.timeit(
    stmt="function1(gkeys, keydict, silent_error)",
    globals=globals(),
    number=100000  # Adjust the number of iterations as needed for more accurate results
)

# Measure execution time for function 2
time_function2 = timeit.timeit(
    stmt="function2(gkeys, keydict, silent_error)",
    globals=globals(),
    number=100000  # Adjust the number of iterations as needed for more accurate results
)

# Print results
print("Execution time for function 1:", time_function1)
print("Execution time for function 2:", time_function2)
