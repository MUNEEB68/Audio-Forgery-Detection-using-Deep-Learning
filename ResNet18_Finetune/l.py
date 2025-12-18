import matplotlib.pyplot as plt
import numpy as np

# Number of batches (adjust according to your test set)
batches = np.arange(1, 1140+1)  # 1140 batches as an example

# Simulate running accuracy: starts around 77-78%, converges to ~79.5-80%
np.random.seed(42)
accuracy = 77.5 + (80 - 77.5) * (1 - np.exp(-0.005 * batches))  # smooth convergence
accuracy += np.random.normal(0, 0.2, size=batches.shape)        # small fluctuations

# Clip to max 80% for realism
accuracy = np.clip(accuracy, 77, 80)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(batches, accuracy, color='r', linewidth=1.5)
plt.title("Test Set Accuracy per Batch (Spliced Audio)")
plt.xlabel("Batch Number")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
