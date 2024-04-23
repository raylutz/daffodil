import timeit

# Option 1: Dict comprehension with enumerated loop
def option1():
    key_col = list(range(1000))  # Sample list of keys
    return {key: index for index, key in enumerate(key_col)}

# Option 2: Using a generator function
def with_index(iterable):
    """Like enumerate, but (val, i) tuples instead of (i, val)."""
    for i, item in enumerate(iterable):
        yield (item, i)

def option2():
    key_col = list(range(1000))  # Sample list of keys
    return dict(with_index(key_col))

# Option 3: Using dict zip
def option3():
    key_col = list(range(1000))  # Sample list of keys
    return dict(zip(key_col, range(len(key_col))))

# Measure execution time for each option
time_option1 = timeit.timeit(option1, number=10000)
time_option2 = timeit.timeit(option2, number=10000)
time_option3 = timeit.timeit(option3, number=10000)

# Print results
print("Option 1 execution time:", time_option1)
print("Option 2 execution time:", time_option2)
print("Option 3 execution time:", time_option3)
