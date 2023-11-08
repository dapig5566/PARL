# PARL: Heterogeneous Order Dispatch for Human-Robot Hybrid Delivery with Preference-Aware Reinforcement Learning
This is the source code and data we used in our paper. 

## Dependencies
The python libraries used in the paper is listed in `requirements.txt`. To install all dependencies, run

```
pip install -r requirements.txt
```

## Data
The data used in our paper is in `data` folder, the data are processed and contains the historical orders for users and delivery. 
Please unzip `dataset.zip` and copy the `.pkl` the files into `data` folder. The paths should be like `data/xxx.pkl`.
For data attributes and details, please refer to [data](https://github.com/dapig5566/PARL/tree/main/data)


## Train
For training, replace the pretrained weight path with your own path in `train.sh` and run 

```
sh ./train.sh
```
For continue a training, please use argument `--ckpt_path <your_ckpt_path>`; For specifing an training model, use argument `--algo <algo_class>`.

## Evaluation   
For evaluation, replace the checkpoint path with your own path in `eval.sh` and run 

```
sh ./eval.sh
```

For evaluating a trained model, argument `--algo` must be specified using the same algo calss as the trained model.