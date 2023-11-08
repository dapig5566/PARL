# PARL: Heterogeneous Order Dispatch for Human-Robot Hybrid Delivery with Preference-Aware Reinforcement Learning
We release the source code and data used in our paper. 

## Dependencies
The python libraries used in the paper is listed in `requirements.txt`. To install all dependencies, run

```
pip install -r requirements.txt
```

## Data
The data used in our paper is in `data` folder, the data are processed and contains the historical orders for users and delivery. 


## Train
For training, replace the pretrained weight path with your own path in `train.sh` and run 

```
sh ./train.sh
```

## Evaluation
For evaluation, replace the checkpoint path with your own path in `eval.sh` and run 

```
sh ./eval.sh
```