# Cross-validation for neural networks.

This repository contains code for Monte Carlo cross-validation or K-fold cross-validation of a CNN-based mitosis classification model. However, the classes for cross-validation can just as easily be used for training other models. 


## Usage

The classes for cross-validation work similarly to the scikit-learn classes, but work with a three-way split into train/val/test. The user simply needs to specify sample_ids, which are split accordingly and can then be used to create different datasets.

```python
from src.splitting import MonteCarloCV

n = 5
val_size = 0.2
test_size = 0.2
n_repeats = 5

samples = np.arange(n)

mccv = MonteCarloCV(val_size, test_size, n_repeats)

for train, val, test in mccv.split(samples):
    print(train, val, test)
    # [2 3 4] [1] [0]
    # [0 2 4] [3] [1]
    # [0 3 4] [2] [1]
    # [1 3 4] [2] [0]
    # [0 1 2] [4] [3]
``` 


```python
from src.splitting import KFoldCV

n = 5
n_folds = 5
kcv = KfoldCV(n_folds, shuffle=False)

samples = np.arange(n)

for train, val, test in kcv.split(samples):
    print(train, val, test)
    # [2 3 4] [0] [1]
    # [0 3 4] [1] [2]
    # [0 1 4] [2] [3]
    # [0 1 2] [3] [4]
    # [1 2 3] [4] [0]
```


## Data
In order to train a mitosis classifier using cross-validation on the MIDOG 2021 dataset, you need to download the images from [Google](https://drive.google.com/drive/folders/1YUMKNkXUtgaFM6jCHpZxIHPZx_CqE_qG) or [Zenodo](https://zenodo.org/record/4643381) and save them under `./image_dir`. The sqlite database is provided under `./annotations/MIDOG.sqlite`.


## Train a Mitosis Classifier with CV

After downloading the data to `./image_dir`, the training of a ResNet18 on MIDOG 2021 using 5-Fold CV and logging everything with Weights & Biases can be performed with the following command:

```
python main.py --image_dir ./image_dir --exp_code KFOLD_5_resnet18 --cv kfoldcv --n_folds 5 --max_epochs 50 --pseudo_epoch_length 1024 --patience 10 --stop_epoch 20 --batch_size 32 --early_stopping --logging --calculate_metrics --shuffle
```