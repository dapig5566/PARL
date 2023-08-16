from main import train_r2d2
from policy_model import DTWrappedRecurrentTDQN
import os

if __name__=="__main__":
    import ray
    from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
    from env import ACTION_SPACE, OBSERVATION_SPACE_LITE_DICT
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    ray.init(num_cpus=8)
    # weight_root = "./weights/pretrain/user_repr_pt_new_data_v2_1678128493"
    weight_root = "./weights/pretrain/TDQN_new_datadist_v4"
    restore_root = "./models/experiment_MVR2D2_ptv3_full_LEN100_1680802237"
    # obj = torch.load(os.path.join(weight_root, "state_dict.pt"))
    # print(obj.__class__)
    # torch.save(model.state_dict(), os.path.join(weight_root, "state_dict.pt"))
    # train("MVDQN_ptv2_org_easy", weight_path=os.path.join(weight_root, "state_dict.pt"))
    # pretrain_embeddings()
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "env_config": {
            "num_cars": 3, 
            "num_couriers": 1,
            "max_episode_length": 300,
            "robot_reward_scale": 2,
            "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
        },
        "num_workers": 6,
        "num_gpus": 1,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": DTWrappedRecurrentTDQN,
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                # "ablation": "user_repr"
            },
            "fcnet_hiddens": [256],
            # "no_final_linear": True,
            # "fcnet_activation": "linear",
            # "use_lstm": True,
            "lstm_cell_size": 256,
            "max_seq_len": 30
        },
        # "target_network_update_freq": 50,

        "zero_init_states": False,
        "min_sample_timesteps_per_iteration": 1000,
        "replay_buffer_config":{
        #   "learning_starts": 1000,
          "type": "MultiAgentReplayBuffer",
          "capacity": 200000,
          "storage_unit": "sequences",
          "replay_burn_in": 0
        },


        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 1000 * 1000,
        },
        "train_batch_size": 64,
        "lr": 5e-4,
        "lr_schedule": [
        [
            0,
            0.0005
        ],
        [
            800000,
            0.0005
        ],
        [
            1500000,
            0.00025
        ],
        [
            3000000,
            0.000125
        ]
    ],
        # "lr_schedule": [[0, 5e-4], [1000*1000, 5e-4], [1000*1800, 2.5e-4], [1000*2500, 1.25e-4]]
    }

    #####################################
    # TRAIN MVR2D2_ptv4_full_FIX300_c3_r
    #####################################
    env_config = {
            "max_episode_length": 300,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": False,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 1,
                "reg_scale": 0.05
            }
    }
    config["env_config"] = env_config
    train_r2d2(
        "MVR2D2_ptv4_full_FIX300_c3_r_0.05reg",
        weight_path=os.path.join(weight_root, "state_dict.pt"),
        config=config,
        iterations=2000,
        restore_path=None
    )
    ray.shutdown()