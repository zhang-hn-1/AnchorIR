# AnchorIR

PyTorch implementation of the AnchorIR image restoration model.

## Requirements

- Python 3.10+
- PyTorch

## Usage

```python
import torch
from model import AnchorIR

model = AnchorIR()
image = torch.rand(1, 3, 256, 256)
restored = model(image)
```
