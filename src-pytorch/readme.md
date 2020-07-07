# Full Body Classification (PyTorch Ver.)

## [1] Requirements
```
to be 
```

## [2] Structure
```
# mainly trained using pytorch-lightning  
src-pytorch/
  └── data/
      ├── train/
          ├── pass/
          └── fail/
      ├── valid/
          ├── pass/
          └── fail/
      └── test/
          ├── pass/
          └── fail/
  ├── experiments/   # for saving best model (pytorch)
  ├── logs/          # for saving checkpoint and hyperparams (pytorch-lightning)
  ├── lightning_engineering.py # pytorch-lightning train code
  ├── lightning_research.py    # pytorch-lightning model code
  ├── dataset.py
  ├── metric.py
  ├── net.py
  ├── test.py 
  ├── train.py 
  ├── utils.py 
  └── readme.md
```

## [3] TODO
- [x] Apply Augmentation
- [x] Get More Data 
- [x] Get More EDGE Data (continuously)
- [x] Add Plotting Code (lightning)
- [x] Add Early Stopping Code (lightning)
- [x] Try PyTorch Lightning Module
- [ ] Flask Demo
