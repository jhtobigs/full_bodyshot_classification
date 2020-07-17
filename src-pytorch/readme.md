# Full Body Classification (PyTorch Ver.)


## Structure
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
  ├── logs/          # for saving checkpoint and hyperparameters (pytorch-lightning)
  ├── lightning_engineering.py  # pytorch-lightning train code
  ├── lightning_research.py     # pytorch-lightning model code (resnet50/mobilenet_v2)
  ├── lightning_efficientnet.py # pytorch-lightning model code (efficientnet-b0)
  ├── lightning_rexnetv1.py     # pytorch-lightning model code (rexnet_v1_1.0)
  ├── dataset.py
  ├── metric.py
  ├── net.py
  ├── train.py 
  ├── utils.py 
  └── readme.md
```

## TODO
- [x] Apply Augmentation
- [x] Get More Data 
- [x] Get More EDGE Data (continuously)
- [x] Add Plotting Code (lightning)
- [x] Add Early Stopping Code (lightning)
- [x] Try PyTorch Lightning Module
- [x] Flask Demo
