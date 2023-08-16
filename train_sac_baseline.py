from main import train_sac
# from policy_model import DTWrappedRecurrentTDQN
import os

if __name__=="__main__":
    import ray
    from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
    from env import ACTION_SPACE, OBSERVATION_SPACE_LITE_DICT
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    ray.init(num_cpus=6)
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
            "max_episode_length": 100,
            # "robot_reward_scale": 3.5,
            # "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 1,
                "reg_scale": 0.05
            }
        },
        "num_workers": 6,
        "num_gpus": 1,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_sac",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "q_model_config": {
            "custom_model": "cdp_qmodel",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "policy_model_config": {
            "custom_model": "cdp_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "optimization": {
            "actor_learning_rate": 2.5e-5,
            "critic_learning_rate": 2.5e-5,
            "entropy_learning_rate": 2.5e-5,
        },
        "train_batch_size": 256,
    }

    train_sac(
        "MVSAC_ptv4_full_RAN100_c3_r", 
        weight_path=os.path.join(weight_root, "state_dict.pt"),
        config=config,
        iterations=2500,
        restore_path=None
    )

    # evaluate(
    #     algo_cls=SAC, 
    #     restore_path="models/experiment_MVSAC_ptv4_full_LEN100_c3_r_1681354510/checkpoint_001056", 
    #     config=config,
    #     recurrent=False
    # )