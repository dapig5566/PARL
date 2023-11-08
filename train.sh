NUM_STEPS=2000
WEIGHT_PATH=weights/pretrain/TDQN_new_datadist_v4
CONFIG_PATH=config/parl.json
MODEL=PARL # possible models: PARL, A2C, SAC, AlphaZero, DQN
EXP_NAME=ptv4_rand_reprod

python run.py \
    --do_train \
    --model $MODEL \
    --exp_name $MODEL-$EXP_NAME \
    --num_steps $NUM_STEPS \
    --weight_path $WEIGHT_PATH \
    --config_path $CONFIG_PATH
