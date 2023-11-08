WEIGHT_PATH=weights/pretrain/TDQN_new_datadist_v4
CKPT_PATH=models/1699238998_PARL_ptv4_rand_newreprod/checkpoint_001360
CONFIG_PATH=config/parl.json

python run.py \
    --do_eval \
    --weight_path $WEIGHT_PATH \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH