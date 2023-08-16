from main import train_a2c
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
        "lr": 2.5e-5,
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "env_config": {
            "num_cars": 3, 
            "num_couriers": 1,
            "max_episode_length": 300,
            "robot_reward_scale": 3.5,
            "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "random_episode": True,
        },
        "num_workers": 6,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        "train_batch_size": 256,
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "entropy_coeff": 0.5,
    }
    train_a2c(
        "MVA2C_ptv4_full_RAN300_c3_r", 
        weight_path=os.path.join(weight_root, "state_dict.pt"),
        config=config,
        iterations=2500,
        restore_path=None
    )
    # evaluate(
    #     algo_cls=A2C, 
    #     restore_path="models/experiment_MVA2C_ptv4_full_LEN100_c3_r_1681540711/checkpoint_000926", 
    #     config=config,
    #     recurrent=False
    # )
    ray.shutdown()