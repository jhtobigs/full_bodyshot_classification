# Full Body Classification (PyTorch Ver.)

## [1] Requirements
```
to be 
```

## [2] Structure
```
src-pytorch/
  └── data/
      ├── train/
          ├── pass/
          └── fail/
      └── valid/
          ├── pass/
          └── fail/
      └── test/
          ├── pass/
          └── fail/
  ├── experiments/ # for saving best model (pytorch)
  ├── logs/ # for saving checkpoint and hyperparams (pytorch-lightning)
  ├── dataset.py
  ├── metric.py
  ├── net.py
  ├── test.py 
  ├── train.py 
  ├── utils.py 
  └── readme.md
```

## [3] TODO
- [ ] Apply Augmentation
- [ ] Get More Data (including EDGE images)
- [ ] Add Plotting Code
- [ ] Add Early Stopping Code
- [x] Try PyTorch Lightning Module
- [ ] Flask Demo