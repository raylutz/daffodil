import csv
import numpy as np
import ast  # To safely evaluate the stored bytes string

# Generate test data: 2 rows with 100-float NumPy arrays
data = [
    [1, "Model_A", np.random.rand(10).astype(np.float32)],
    [2, "Model_B", np.random.rand(10).astype(np.float32)]
]

# Step 1: Write to CSV
with open("test_weights.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name", "weights"])  # Header
    writer.writerows(data)

print("✅ Data written to CSV.")

# Step 2: Read from CSV and reconstruct NumPy arrays
with open("test_weights.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    reconstructed_data = []
    for row in reader:
        # Convert the stored string back to bytes using ast.literal_eval()
        weights_bytes = ast.literal_eval(row[2])  
        # Reconstruct the original NumPy array
        weights_array = np.frombuffer(weights_bytes, dtype=np.float32)
        # breakpoint()
        reconstructed_data.append([int(row[0]), row[1], weights_array])

print("✅ Data read from CSV and reconstructed.")

# Step 3: Verify round-trip integrity
for orig, recon in zip(data, reconstructed_data):
    assert orig[0] == recon[0], "ID mismatch!"
    assert orig[1] == recon[1], "Name mismatch!"
    assert np.allclose(np.frombuffer(orig[2], dtype=np.float32), recon[2]), "Weights mismatch!"

print("✅ Round-trip test successful: Original and reconstructed data match.")

# Step 4: Display sample results
print("\nSample reconstructed data:")
for row in reconstructed_data:
    print(f"ID: {row[0]}, Name: {row[1]}, First 5 Weights: {row[2][:5]}")
