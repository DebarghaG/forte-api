# Quickstart

## Install

```bash
pip install forte-detector
```

## Train

```python
from forte import ForteOODDetector

detector = ForteOODDetector(method='gmm', device='cuda:0')
detector.fit(train_image_paths)
```

First run downloads ~2GB of pretrained models.

## Predict

```python
predictions = detector.predict(test_paths)   # 1=ID, -1=OOD
scores = detector.predict_proba(test_paths)  # [0,1], higher=ID
```

## Evaluate

```python
metrics = detector.evaluate(id_test_paths, ood_test_paths)
print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR@95: {metrics['FPR@95TPR']:.4f}")
```

## Full Example

```python
import glob
from forte import ForteOODDetector

# Collect image paths
id_train = glob.glob("data/normal/train/*.jpg")
id_test = glob.glob("data/normal/test/*.jpg")
ood_test = glob.glob("data/anomaly/test/*.jpg")

# Train and evaluate
detector = ForteOODDetector(method='gmm', device='cuda:0')
detector.fit(id_train)
metrics = detector.evaluate(id_test, ood_test)

print(metrics)
```

## Next

- [Algorithm](methods.md) - How it works
- [API Reference](api-reference.md) - Full API
- [Configuration](user-guide.md) - Parameters
