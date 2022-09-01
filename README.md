# Codes for transfer learning on material data


## Data

Most data we tried is in the `data` folder. `data/data_split_5_percent.pkl` uses only 5% percent high fidelity data as the training set. `data/data_split_10_percent.pkl` uses 10% percent high fidelity data as the training set.
## Experiments

### Pretrain

```python
python pretrain.py
```

### Our methods
```
python finetune.py --lower-steps 1 --reg-lr 3.5 --reg-lamb 3e-3 
```

### Direct Fine-tuning
```
python finetune.py --lower-steps 0 --reg-lr 3.5 --reg-lamb 3e-3  --no-alpha --no-beta 
```

### No Pretraining
```
python finetune.py --lower-steps 0 --reg-lr 3.5 --reg-lamb 3e-3  --no-alpha --no-beta --no-pretrain
```

### Reproduced Results

5 seeds average:

our method: 0.0913

direct fine-tuning: 0.153574138879776

no pretraining: 0.2183179408311844


