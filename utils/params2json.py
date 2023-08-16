import json

default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v3",
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
        "num_workers": 4,
        "num_gpus": 1,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "WrappedRecurrentDQN",
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

env_config = {
            "num_cars": 3, 
            "dataset_index": 0,
            "num_couriers": 1,
            "max_episode_length": 100,
            "robot_reward_scale": 2,
            "entropy_reward_scale": 2.5,
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

config = {
    "default_config": default_config,
    "env_config": env_config
}

with open("config/pdr2d2.conf", "w") as f:
    json.dump(config, f)