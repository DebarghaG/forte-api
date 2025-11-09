# Examples

Real-world examples of using Forte for out-of-distribution detection.

## Table of Contents

1. [CIFAR-10 vs CIFAR-100](#cifar-10-vs-cifar-100)
2. [Custom Image Dataset](#custom-image-dataset)
3. [Medical Imaging](#medical-imaging-anomaly-detection)
4. [Quality Control](#manufacturing-quality-control)
5. [Multi-Method Comparison](#comparing-detection-methods)

---

## CIFAR-10 vs CIFAR-100

Detect CIFAR-100 images as out-of-distribution when trained on CIFAR-10.

```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from forte import ForteOODDetector

# Download datasets
transform = transforms.ToTensor()
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Save as PNG files
def save_dataset(dataset, save_dir, num_images=1000):
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        path = os.path.join(save_dir, f"{i}.png")
        image.save(path)
        paths.append(path)
    return paths

id_train = save_dataset(cifar10_train, "data/cifar10/train", 5000)
id_test = save_dataset(cifar10_test, "data/cifar10/test", 1000)
ood_test = save_dataset(cifar100_test, "data/cifar100/test", 1000)

# Train detector
detector = ForteOODDetector(method='gmm', device='cuda:0' if torch.cuda.is_available() else 'cpu')
detector.fit(id_train)

# Evaluate
metrics = detector.evaluate(id_test, ood_test)
print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR@95TPR: {metrics['FPR@95TPR']:.4f}")
```

---

## Custom Image Dataset

Use Forte with your own image dataset.

```python
import os
from pathlib import Path
from forte import ForteOODDetector

# Organize your images
data_dir = Path("/path/to/your/data")

# Collect image paths
id_train_paths = sorted(list((data_dir / "normal" / "train").glob("*.jpg")))
id_test_paths = sorted(list((data_dir / "normal" / "test").glob("*.jpg")))
ood_test_paths = sorted(list((data_dir / "anomalous" / "test").glob("*.jpg")))

print(f"Training images: {len(id_train_paths)}")
print(f"ID test images: {len(id_test_paths)}")
print(f"OOD test images: {len(ood_test_paths)}")

# Create detector
detector = ForteOODDetector(
    method='gmm',
    nearest_k=5,
    batch_size=32,
    device='cuda:0',
    embedding_dir='./cache'
)

# Train
print("Training detector...")
detector.fit(id_train_paths, val_split=0.2)

# Get predictions
print("Making predictions...")
test_paths = id_test_paths + ood_test_paths
predictions = detector.predict(test_paths)
scores = detector.predict_proba(test_paths)

# Analyze results
id_correct = (predictions[:len(id_test_paths)] == 1).mean()
ood_correct = (predictions[len(id_test_paths):] == -1).mean()

print(f"ID detection rate: {id_correct:.2%}")
print(f"OOD detection rate: {ood_correct:.2%}")

# Evaluate
metrics = detector.evaluate(id_test_paths, ood_test_paths)
print(f"\\nMetrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
```

---

## Medical Imaging Anomaly Detection

Detect anomalous medical scans.

```python
from pathlib import Path
from forte import ForteOODDetector
import matplotlib.pyplot as plt
import numpy as np

# Load medical images
# Assume we have normal X-rays and abnormal (tumor) X-rays
normal_train = list(Path("data/medical/normal/train").glob("*.png"))
normal_test = list(Path("data/medical/normal/test").glob("*.png"))
abnormal_test = list(Path("data/medical/abnormal/test").glob("*.png"))

# Create detector optimized for medical images
detector = ForteOODDetector(
    method='gmm',        # GMM works well for medical images
    nearest_k=10,        # Higher k for more robust PRDC
    batch_size=16,       # Smaller batches for large images
    device='cuda:0'
)

# Train on normal scans only
detector.fit(normal_train, val_split=0.15)

# Evaluate
metrics = detector.evaluate(normal_test, abnormal_test)

print("Medical Imaging OOD Detection Results:")
print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR@95TPR: {metrics['FPR@95TPR']:.4f}")

# Get scores for visualization
normal_scores = detector.predict_proba(normal_test)
abnormal_scores = detector.predict_proba(abnormal_test)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
plt.hist(abnormal_scores, bins=50, alpha=0.7, label='Abnormal', density=True)
plt.xlabel('Normality Score')
plt.ylabel('Density')
plt.title('Medical Image Anomaly Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('medical_ood_results.png')

# Find threshold for 95% sensitivity on normal scans
threshold = np.percentile(normal_scores, 5)
sensitivity = (abnormal_scores < threshold).mean()
print(f"\\nAt 95% specificity:")
print(f"  Threshold: {threshold:.4f}")
print(f"  Abnormality detection rate: {sensitivity:.2%}")
```

---

## Manufacturing Quality Control

Detect defective products on a production line.

```python
from forte import ForteOODDetector
from pathlib import Path
import time

# Paths to product images
good_products_train = list(Path("data/factory/good/train").glob("*.jpg"))
good_products_test = list(Path("data/factory/good/test").glob("*.jpg"))
defective_products = list(Path("data/factory/defective/test").glob("*.jpg"))

print(f"Training on {len(good_products_train)} good product images...")

# Create fast detector for real-time inspection
detector = ForteOODDetector(
    method='ocsvm',      # Fast method for production
    nearest_k=5,
    batch_size=64,       # Large batches for speed
    device='cuda:0'
)

# Train
start_time = time.time()
detector.fit(good_products_train, val_split=0.1)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate accuracy
metrics = detector.evaluate(good_products_test, defective_products)
print(f"\\nQuality Control Performance:")
print(f"  AUROC: {metrics['AUROC']:.4f}")
print(f"  False Alarm Rate @95% Detection: {metrics['FPR@95TPR']:.2%}")

# Test inference speed
test_batch = good_products_test[:100]
start_time = time.time()
predictions = detector.predict(test_batch)
inference_time = (time.time() - start_time) / len(test_batch)
print(f"\\nInference Performance:")
print(f"  Time per image: {inference_time*1000:.2f} ms")
print(f"  Throughput: {1/inference_time:.1f} images/second")

# Real-time inspection simulation
def inspect_product(image_path):
    """Simulate real-time product inspection."""
    score = detector.predict_proba([image_path])[0]
    threshold = 0.5  # Adjust based on requirements
    is_good = score > threshold
    return is_good, score

# Test on new products
for product_path in good_products_test[:5]:
    is_good, score = inspect_product(product_path)
    status = "PASS" if is_good else "FAIL"
    print(f"{product_path.name}: {status} (score: {score:.3f})")
```

---

## Comparing Detection Methods

Compare GMM, KDE, and OCSVM on the same dataset.

```python
from forte import ForteOODDetector
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_paths = [...]  # Your training data
id_test_paths = [...]  # Your ID test data
ood_test_paths = [...]  # Your OOD test data

# Test all methods
methods = ['gmm', 'kde', 'ocsvm']
results = {}

for method in methods:
    print(f"\\nTesting {method.upper()}...")

    detector = ForteOODDetector(
        method=method,
        device='cuda:0',
        embedding_dir=f'./cache_{method}'
    )

    # Train
    detector.fit(train_paths)

    # Evaluate
    metrics = detector.evaluate(id_test_paths, ood_test_paths)
    results[method] = metrics

    print(f"  AUROC: {metrics['AUROC']:.4f}")
    print(f"  FPR@95TPR: {metrics['FPR@95TPR']:.4f}")

# Create comparison table
df = pd.DataFrame(results).T
print("\\nComparison Table:")
print(df.to_string())

# Plot comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
metrics_names = ['AUROC', 'FPR@95TPR', 'AUPRC', 'F1']

for ax, metric in zip(axes, metrics_names):
    values = [results[m][metric] for m in methods]
    ax.bar(methods, values)
    ax.set_title(metric)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('method_comparison.png')
print("\\nComparison plot saved to 'method_comparison.png'")
```

---

## Batch Processing for Large Datasets

Efficiently process large numbers of images.

```python
from forte import ForteOODDetector
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Large dataset
all_test_images = list(Path("data/large_dataset").rglob("*.jpg"))
print(f"Processing {len(all_test_images)} images...")

# Create detector
detector = ForteOODDetector(
    method='gmm',
    batch_size=128,  # Large batch for efficiency
    device='cuda:0'
)

# Train
detector.fit(train_paths)

# Process in chunks to manage memory
chunk_size = 1000
all_scores = []

for i in tqdm(range(0, len(all_test_images), chunk_size)):
    chunk = all_test_images[i:i + chunk_size]
    scores = detector.predict_proba(chunk)
    all_scores.extend(scores)

all_scores = np.array(all_scores)

# Analyze results
threshold = 0.5
num_ood = (all_scores < threshold).sum()
print(f"\\nResults:")
print(f"  Total images: {len(all_scores)}")
print(f"  Detected as OOD: {num_ood} ({num_ood/len(all_scores):.1%})")
print(f"  Mean score: {all_scores.mean():.3f}")
print(f"  Std score: {all_scores.std():.3f}")

# Save results
results_df = pd.DataFrame({
    'image_path': [str(p) for p in all_test_images],
    'score': all_scores,
    'is_ood': all_scores < threshold
})
results_df.to_csv('ood_detection_results.csv', index=False)
print("Results saved to 'ood_detection_results.csv'")
```

---

## Custom Thresholding

Set custom detection thresholds based on your requirements.

```python
from forte import ForteOODDetector
import numpy as np
from sklearn.metrics import precision_recall_curve

# Train detector
detector = ForteOODDetector()
detector.fit(train_paths)

# Get scores
id_scores = detector.predict_proba(id_test_paths)
ood_scores = detector.predict_proba(ood_test_paths)

# Combine for threshold selection
all_scores = np.concatenate([id_scores, ood_scores])
all_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)

# Strategy 1: Maximize F1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]
print(f"Best F1 threshold: {best_f1_threshold:.3f} (F1={f1_scores[best_f1_idx]:.3f})")

# Strategy 2: High recall (95%)
high_recall_idx = np.where(recall >= 0.95)[0][0]
high_recall_threshold = thresholds[high_recall_idx]
print(f"95% recall threshold: {high_recall_threshold:.3f} (precision={precision[high_recall_idx]:.3f})")

# Strategy 3: High precision (95%)
high_precision_idx = np.where(precision >= 0.95)[0][-1]
high_precision_threshold = thresholds[high_precision_idx]
print(f"95% precision threshold: {high_precision_threshold:.3f} (recall={recall[high_precision_idx]:.3f})")

# Apply custom threshold
def detect_with_threshold(image_paths, threshold):
    scores = detector.predict_proba(image_paths)
    return np.where(scores > threshold, 1, -1)

# Test with different thresholds
for name, thresh in [("Best F1", best_f1_threshold),
                      ("High Recall", high_recall_threshold),
                      ("High Precision", high_precision_threshold)]:
    preds = detect_with_threshold(ood_test_paths, thresh)
    ood_detection_rate = (preds == -1).mean()
    print(f"{name}: OOD detection rate = {ood_detection_rate:.2%}")
```

---

## Next Steps

- [Methods](methods.md) - Understand the algorithms
- [User Guide](user-guide.md) - Learn advanced features
- [API Reference](api-reference.md) - Detailed API documentation
