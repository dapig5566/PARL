WEIGHT_PATH=weights/pretrain/TDQN_new_datadist_v4
CKPT_PATH=models/1699238998_MVR2D2_ptv4_full_RAN300_c3_r_newreprod/checkpoint_001360
CONFIG_PATH=config/parl.json
MODEL=PARL # the same class as it is in ckpt_path. possible models: A2C, SAC, AlphaZero, DQN

python run.py \
    --do_eval \
    --model $MODEL \
    --weight_path $WEIGHT_PATH \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH