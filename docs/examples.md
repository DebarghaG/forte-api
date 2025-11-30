# Examples

## CIFAR-10 vs CIFAR-100

```python
import os
import torch
import torchvision
from torchvision import transforms
from forte import ForteOODDetector

def save_images(dataset, path, n=1000):
    os.makedirs(path, exist_ok=True)
    paths = []
    for i in range(min(n, len(dataset))):
        img, _ = dataset[i]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        p = os.path.join(path, f"{i}.png")
        img.save(p)
        paths.append(p)
    return paths

cifar10_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
cifar10_test = torchvision.datasets.CIFAR10('./data', train=False, download=True)
cifar100_test = torchvision.datasets.CIFAR100('./data', train=False, download=True)

id_train = save_images(cifar10_train, 'data/c10/train', 5000)
id_test = save_images(cifar10_test, 'data/c10/test', 1000)
ood_test = save_images(cifar100_test, 'data/c100/test', 1000)

detector = ForteOODDetector(method='gmm', device='cuda:0')
detector.fit(id_train)
print(detector.evaluate(id_test, ood_test))
```

## Custom Dataset

```python
from pathlib import Path
from forte import ForteOODDetector

id_train = list(Path('data/normal/train').glob('*.jpg'))
id_test = list(Path('data/normal/test').glob('*.jpg'))
ood_test = list(Path('data/anomaly').glob('*.jpg'))

detector = ForteOODDetector(method='gmm')
detector.fit([str(p) for p in id_train])
print(detector.evaluate([str(p) for p in id_test], [str(p) for p in ood_test]))
```

## Method Comparison

```python
from forte import ForteOODDetector

results = {}
for method in ['gmm', 'kde', 'ocsvm']:
    det = ForteOODDetector(method=method, embedding_dir=f'./cache_{method}')
    det.fit(train_paths)
    results[method] = det.evaluate(id_test, ood_test)

for m, r in results.items():
    print(f"{m}: AUROC={r['AUROC']:.4f} FPR@95={r['FPR@95TPR']:.4f}")
```
