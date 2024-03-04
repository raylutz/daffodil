import timeit

# Sample list of keys
key_col = [f"row{i}" for i in range(1000)]

# Option 1: Dict comprehension with enumerated loop
def option1():
    return {key: index for index, key in enumerate(key_col)}

# Option 2: Using a generator function
def with_index(iterable):
    """Like enumerate, but (val, i) tuples instead of (i, val)."""
    for i, item in enumerate(iterable):
        yield (item, i)

def option2():
    return dict(with_index(key_col))

# Option 3: Using dict zip
def option3():
    return dict(zip(key_col, range(len(key_col))))

# Measure execution time for each option
time_option1 = timeit.timeit(option1, number=10000)
time_option2 = timeit.timeit(option2, number=10000)
time_option3 = timeit.timeit(option3, number=10000)

# Print results
print("Option 1 execution time:", time_option1)
print("Option 2 execution time:", time_option2)
print("Option 3 execution time:", time_option3)

percent_slower_option1 = ((time_option1 - time_option3) / time_option3) * 100
percent_slower_option2 = ((time_option2 - time_option3) / time_option3) * 100

print("Option 1 is {:.2f}% slower than Option 3.".format(percent_slower_option1))
print("Option 2 is {:.2f}% slower than Option 3.".format(percent_slower_option2))
