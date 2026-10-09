# PA_CAE

Passenger counting code and the XiamenBus dataset.

## Downloads

[Download data and weights](https://github.com/AIOTGzg/PA-CAE/releases/tag/v1.0.0).

The dataset contains **886 training images and 222 test images**, with MAT annotations, CSV density maps, and split lists.

## Setup

```bash
pip install -r requirements.txt
```

Extract `XiamenBus.zip` to obtain `XiamenBus/train_data`, `XiamenBus/test_data`, and `XiamenBus/splits`.

## Training

```bash
python train.py --data-root /path/to/XiamenBus --device cuda --epochs 200 --batch-size 8 --imagenet-pretrained --output runs/pa_cae
```

## Evaluation

For a trained PA_CAE checkpoint:

```bash
python evaluate.py --data-root /path/to/XiamenBus --weights runs/pa_cae/pa_cae_latest.pth --device cuda --output predictions.csv
```

The supplied `weights.pth` requires the separate `resnet101` model; it is incompatible with `PA_CAE`.

```bash
python evaluate.py --data-root /path/to/XiamenBus --model resnet101 --weights weights.pth --device cuda --output predictions.csv
```

Results include per-image counts and absolute errors, MAE, RMSE, and threshold statistics.
