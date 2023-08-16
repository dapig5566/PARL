# Import the RL algorithm (Algorithm) we would like to use.

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

# import ray
# NUM_MAX_CPUS = 5
# ray.init(ignore_reinit_error=True, num_cpus=NUM_MAX_CPUS)

from ray.rllib.algorithms.a2c import A2C
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.alpha_zero import AlphaZero
from ray.rllib.algorithms.qmix import QMix
from ray.rllib.algorithms.r2d2 import R2D2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from multi_agent_env import MockEnvWithGroupedAgents
from algo import CDPModel, CDPModel_DQNver, CDP_QModel
from policy_model import CDPModel_DQN_with_pretrain, CDPModel_AlphaZero, DTWrappedRecurrentTDQN
from rdqn_v3 import WrappedRecurrentDQN
from cdp_sac import CDPSACTorchModel
from env import OrderEnv2
from env_v3 import OrderEnv3
# from ray.rllib.examples.env.parametric_actions_cartpole import ParametricActionsCartPole
# from ray.rllib.examples.models.parametric_actions_model import (
#     ParametricActionsModel,
#     TorchParametricActionsModel,
# )
from config import NUM_ITERS, NUM_CARS
import json
import torch
import numpy as np
import os
import time
import sys
import pickle as pkl
from tqdm import tqdm

ModelCatalog.register_custom_model("cdp_model", CDPModel)
ModelCatalog.register_custom_model("cdp_model_dqnver", CDPModel_DQNver)
ModelCatalog.register_custom_model("cdp_qmodel", CDP_QModel)
ModelCatalog.register_custom_model("cdp_sac", CDPSACTorchModel)
ModelCatalog.register_custom_model("cdp_dqn_pt", CDPModel_DQN_with_pretrain)
ModelCatalog.register_custom_model("cdp_az", CDPModel_AlphaZero)
ModelCatalog.register_custom_model("WrappedRecurrentDQN", WrappedRecurrentDQN)
# ModelCatalog.register_custom_model("cdp_r2d2", RecurrentTDQN)

# ModelCatalog.register_custom_model("pa_model",TorchParametricActionsModel)
register_env("OrderEnv-v2", OrderEnv2)
register_env("OrderEnv-v3", OrderEnv3)

# register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))
# Configure the algorithm.

def save_ckpt(algo, exp_name, num_ckpt=5, config=None):
    algo.save("models/{}".format(exp_name))

    if config is not None:
        if not os.path.exists(os.path.join("models", exp_name, "params.json")):
            try:
                with open(os.path.join("models", exp_name, "params.json"), "w") as f:
                    json.dump(conf, f)
            
            except:
                with open(os.path.join("models", exp_name, "params.json"), "w") as f:
                    f.write(str(config))

    ckpt_path = os.path.join(".", "models", exp_name)
    ckpt_files = sorted(os.listdir(ckpt_path))
    if len(ckpt_files) > 5:
        for file in ckpt_files[:-num_ckpt]:
            os.system("rm -r {}".format(os.path.join(ckpt_path, file)))

    

def train(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 8,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_dqn_pt",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        # "entropy_coeff": 0.05,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 2000 * 2000,
        },
        "train_batch_size": 128,
        "lr": 2.5e-4,
        "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.
    # algo = A2C(env="OrderEnv-v0", config=config)

    algo = DQN(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        # policy_info = result["info"]["learner"]["default_policy"]["learner_stats"]
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        
        if mean_reward > max_reward or i == iterations - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
        bar.update(1)
        # results.append(result)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()

    # pkl.dump(results, open("models\\experiment_{}\\results.dat".format(timestamp), "wb"))


def train_alpha_zero(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    
    default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_rollout_workers": 2,
        
        
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_az",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "replay_buffer_config": {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 1000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, episodes or timesteps.
            "storage_unit": "fragments",
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "lr": 5e-5,
        # "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.

    algo = AlphaZero(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        # policy_info = result["info"]["learner"]["default_policy"]["learner_stats"]
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        
        if mean_reward > max_reward or i == iterations - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
        bar.update(1)
        # results.append(result)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()


def train_a2c(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    default_config = {
        "lr": 2.5e-5,
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 20,
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
        "entropy_coeff": 0.05,
        # "exploration_config" : {
        #     "type": "EpsilonGreedy",
        #     "initial_epsilon": 1.0,
        #     "final_epsilon": 0.1,
        #     "epsilon_timesteps": 1500*1000,
        # },
        # "hiddens": [64, ],
        # "dueling": False,
        # "evaluation_num_workers": 1,
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        # "disable_env_checking": True
        # Only for evaluation runs, render the env.
    }
    if config is None:
        config = default_config
    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.
    # algo = A2C(env="OrderEnv-v0", config=config)
    algo = A2C(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        policy_info = result["info"]["learner"]["default_policy"]["learner_stats"]
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        if mean_reward > max_reward or i == NUM_ITERS - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
            
        bar.update(1)
        # results.append(result)

        # jstr = str(result)
        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()

    # pkl.dump(results, open("models\\experiment_{}\\results.dat".format(timestamp), "wb"))


def train_sac(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 20,
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
        # "entropy_coeff": 0.05,
        # "exploration_config" : {
        #     "type": "EpsilonGreedy",
        #     "initial_epsilon": 1.0,
        #     "final_epsilon": 0.02,
        #     "epsilon_timesteps": NUM_ITERS//2*1000,
        # },
        # "hiddens": [64, ],
        # "dueling": False,
        # "evaluation_num_workers": 1,
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        # "disable_env_checking": True
        # Only for evaluation runs, render the env.
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.
    # algo = A2C(env="OrderEnv-v0", config=config)
    algo = SAC(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        if mean_reward > max_reward or i == NUM_ITERS - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
        # results.append(result)
        bar.update(1)
        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()

    # pkl.dump(results, open("models\\experiment_{}\\results.dat".format(timestamp), "wb"))


def evaluate(algo_cls, restore_path, config, recurrent=False, env=None):
    from ray.rllib.policy.sample_batch import SampleBatch
    import tree
    import torch
    raw_config = {
        "lr": 2.5e-5,
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "entropy_coeff": 0.05,
        # "exploration_config" : {
        #     "type": "EpsilonGreedy",
        #     "initial_epsilon": 1.0,
        #     "final_epsilon": 0.1,
        #     "epsilon_timesteps": 1500*1000,
        # },
        # "hiddens": [64, ],
        # "dueling": False,
        # "evaluation_num_workers": 1,
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        # "disable_env_checking": True
        # Only for evaluation runs, render the env.
    }
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #

    # Create our RLlib Trainer.
    # algo = A2C(env="OrderEnv-v0", config=config)
    algo = algo_cls(config=config)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    dist_dicts = []
    results = []
    statistics = []
    maps = []
    routes = []
    q_values = []
    algo.restore(restore_path)

    if env is None:
        env = OrderEnv2(config["env_config"])
    
    model = algo.get_policy().model
    for ep in tqdm(range(15)):
        episode_return = 0

        ob, _ = env.reset()
        state = model.get_initial_state() if recurrent else []
        
        for i in range(env.spec.max_episode_steps):
            result = algo.compute_single_action(
                observation=ob,
                state=state,
                explore=False,
                policy_id="default_policy",  # <- default value
            )
            
            if isinstance(algo, (A2C, SAC)):
                action = result
            else:
                action, state, _ = result
            
            # input_dict = {SampleBatch.OBS: ((torch.from_numpy(np.array(ob[0])).unsqueeze(0).float(),
            #                                 torch.from_numpy(np.array(ob[1])).unsqueeze(0).float(),
            #                                 torch.from_numpy(np.array(ob[2])).unsqueeze(0)) )}
            # emb, _ = model(input_dict, [], 0)
            # q_value = model.get_q_value_distributions(emb)
            # q_values.append((q_value[0].detach().numpy(), env.next_levels[env.steps]))

            ob, r, d, _, _ = env.step(action)
            episode_return += r
        dist_dict = env.show_order_dist()
        dist_dicts.append(dist_dict)
        results.append(episode_return)
        statistics.append(
            [*env.car_dispatched_num, 
             *env.car_assigned_num, 
             env.courier_dispatched_num, 
             env.courier_assigned_num])
        maps.append(env.dispatch_map)
        routes.append(env.car_routes)

    print(results)
    print(np.mean(results))
    for i in statistics:
        print(i)
    mean_stat = np.mean(statistics, axis=0)

    print("mean statistics: {}".format(mean_stat))
    print()
    print("car dispatch rate: {}/{}: {}".format(sum(mean_stat[0:NUM_CARS]),
                                                sum(mean_stat[NUM_CARS:2*NUM_CARS]),
                                                sum(mean_stat[0:NUM_CARS]) / sum(mean_stat[NUM_CARS:2*NUM_CARS])))
    print()
    for i in range(NUM_CARS):
        print("car_{} dispatch rate: {}/{}: {}".format(i, mean_stat[i],
                                                       mean_stat[i + NUM_CARS],
                                                       mean_stat[i] / mean_stat[i + NUM_CARS]))
    print()
    print("courier dispatch rate: {}/{}, {}".format(mean_stat[-2],
                                                    mean_stat[-1],
                                                    mean_stat[-2] / mean_stat[-1]))
    print()
    courier_dist = np.sum([dc["courier"] for dc in dist_dicts], axis=0)
    courier_dist = courier_dist / np.sum(courier_dist)

    car_dists = np.sum([dc["car"] for dc in dist_dicts], axis=0)
    car_dists = car_dists / np.sum(car_dists)

    print(f"courier order distribution: {courier_dist}")
    print(f"car order distribution: {car_dists}")
    print()
    print(f"dispatch ratio: {sum(mean_stat[NUM_CARS: 2*NUM_CARS])/mean_stat[-1]}")
    print(f"workload std: {np.std(mean_stat[NUM_CARS: 2*NUM_CARS])}")
    print(f"car entropy mean: {np.mean(env.disp_car_ents)}, std: {np.std(env.disp_car_ents)}")
    print(f"courier entropy mean: {np.mean(env.disp_courier_ents)}, std: {np.std(env.disp_courier_ents)}")
    return maps, routes
    # print("Q Values:")
    # for i in range(20):
    #     print(q_values[i])


def pretrain_embeddings():
    from policy_model import CDPModel_DQN_with_pretrain, CDPModel_TDQN_with_pretrain
    from env import OrderEnv2
    from order_dataset import OrderDataset
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import StepLR
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    timestp = int(time.time())
    model_name = "DQN_full_datadist_ptv3"
    dir_path = "weights/pretrain/{}_{}".format(model_name, timestp)
    file_name = "state_dict.pt"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    _device = torch.device("cuda:0")
    env = OrderEnv2()

    model = CDPModel_TDQN_with_pretrain(obs_space=env.observation_space,
                                       action_space=env.action_space,
                                       num_outputs=256,
                                       model_config={},
                                       name="CDP_TDQN_PRETRAIN",
                                       pre_train=True, device=_device, recurrent=False
                                       ).to(_device)

    clsf = nn.Linear(256, 7).to(_device)
    level_est = nn.Linear(256, 5).to(_device)

    xent_loss_fn = nn.CrossEntropyLoss()

    # optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    lr_sched = StepLR(optim, step_size=50, gamma=0.5)
    train_dataset = OrderDataset(0, split="train", max_length=20000)
    data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_data_loader = DataLoader(OrderDataset(0, max_length=2000, user_list=train_dataset.get_user_list(), split="valid"), batch_size=128, shuffle=True)

    max_acc = -np.inf
    for i in range(50):
        avg_loss = 0
        avg_acc = 0
        avg_lvl_acc = 0
        model.train()
        for x, y1, y2 in tqdm(data_loader):
            x = [t.to(_device) for t in x]
            y1 = y1.to(_device)
            y2 = y2.to(_device)
            input_dict = {"obs": [x]}
            feature, _ = model(input_dict)
            logits = clsf(feature)
            lvl_pred = level_est(feature)
            cls_loss = xent_loss_fn(logits, y1)
            # cls_loss = 0
            lvl_loss = xent_loss_fn(lvl_pred, y2)
            # lvl_loss = 0
            loss = cls_loss + lvl_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss = avg_loss + loss.detach().item()
            avg_acc  = avg_acc + torch.mean(torch.eq(torch.argmax(logits, dim=-1), torch.argmax(y1, dim=-1)).float()).detach().item()
            avg_lvl_acc  = avg_lvl_acc + torch.mean(torch.eq(torch.argmax(lvl_pred, dim=-1), torch.argmax(y2, dim=-1)).float()).detach().item()
        lr_sched.step()
        print("EP{}: loss: {:.5f}, acc: {:.3f}, lvl_acc: {:.3f}".format(i, avg_loss/len(data_loader), avg_acc/len(data_loader), avg_lvl_acc/len(data_loader)))

        model.eval()
        avg_eval_loss = 0
        avg_eval_acc = 0
        avg_eval_lvl_acc = 0
        avg_eval_hard_acc = 0
        avg_eval_easy_acc = 0
        for eval_x, eval_y1, eval_y2 in eval_data_loader:
            eval_x = [t.to(_device) for t in eval_x]
            eval_y1 = eval_y1.to(_device)
            eval_y2 = eval_y2.to(_device)
            input_dict = {"obs": [eval_x]}
            feature, _ = model(input_dict)
            logits = clsf(feature)
            lvl_pred = level_est(feature)
            cls_loss = xent_loss_fn(logits, eval_y1)
            # cls_loss = 0
            lvl_loss = xent_loss_fn(lvl_pred, eval_y2)
            # lvl_loss = 0
            loss = cls_loss + lvl_loss

            acc = torch.mean(torch.eq(torch.argmax(logits, dim=-1), torch.argmax(eval_y1, dim=-1)).float()).detach().item()
            lvl_acc = torch.mean(torch.eq(torch.argmax(lvl_pred, dim=-1), torch.argmax(eval_y2, dim=-1)).float()).detach().item()
            

            # hard_ind = [i for i, v in enumerate(torch.argmax(eval_y1, dim=-1).detach().cpu().numpy()) if v == 7]
            # hard_logits = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            # hard_logits = np.array([hard_logits[i] for i in hard_ind])
            # hard_acc = sum(hard_logits == 7)/len(hard_ind)

            # easy_ind = [i for i, v in enumerate(torch.argmax(eval_y1, dim=-1).detach().cpu().numpy()) if v != 7] 
            # easy_logits = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            # easy_logits = np.array([easy_logits[i] for i in easy_ind])
            # easy_label = torch.argmax(eval_y1, dim=-1).detach().cpu().numpy()
            # easy_label = np.array([easy_label[i] for i in easy_ind])
            # easy_acc = sum(easy_logits == easy_label)/len(easy_ind)

            avg_eval_loss = avg_eval_loss + loss
            avg_eval_acc = avg_eval_acc + acc
            avg_eval_lvl_acc = avg_eval_lvl_acc + lvl_acc
            # avg_eval_hard_acc = avg_eval_hard_acc + hard_acc
            # avg_eval_easy_acc = avg_eval_easy_acc + easy_acc
        print("EP{} (VAL): loss: {:.5f}, acc: {:.5f}, lvl_acc: {:.5f}".format(i, avg_eval_loss/len(eval_data_loader), avg_eval_acc/len(eval_data_loader), avg_eval_lvl_acc/len(eval_data_loader)))
        if avg_eval_acc/len(eval_data_loader) > max_acc:
            torch.save(model.state_dict(), os.path.join(dir_path, file_name))
            print("New Model Saved.")
            max_acc = avg_eval_acc/len(eval_data_loader)
    
    torch.save(model.state_dict(), os.path.join(dir_path, file_name))



def train_qmix_test(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    
    default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": MockEnvWithGroupedAgents,
        "num_workers": 0,
        
        
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        # "model": {
        #     "custom_model": "cdp_az",
        #     # Extra kwargs to be passed to your model's c'tor.
        #     "custom_model_config": {
        #     },
        # },

        # "replay_buffer_config": {
        #     "type": "ReplayBuffer",
        #     # Size of the replay buffer in batches (not timesteps!).
        #     "capacity": 1000,
        #     # Choosing `fragments` here makes it so that the buffer stores entire
        #     # batches, instead of sequences, episodes or timesteps.
        #     "storage_unit": "fragments",
        # },

        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            # Timesteps over which to anneal epsilon.
            "epsilon_timesteps": 40000,

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        "train_batch_size": 64,
        
        "lr": 2.5e-4,
        # "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.

    algo = QMix(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        # policy_info = result["info"]["learner"]["default_policy"]["learner_stats"]
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        
        if mean_reward > max_reward or i == iterations - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
        bar.update(1)
        # results.append(result)
        # print(str(mean_reward))
        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()


def train_r2d2(exp_name, weight_path=None, restore_path=None, iterations=NUM_ITERS, config=None):
    default_config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 8,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_dqn_pt",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        # "entropy_coeff": 0.05,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 2000 * 2000,
        },
        "train_batch_size": 128,
        "lr": 2.5e-4,
        "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    #
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    # Create our RLlib Trainer.
    # algo = A2C(env="OrderEnv-v0", config=config)

    algo = R2D2(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.user_embedding.load_state_dict(state_dict)
    
    if restore_path is not None:
        algo.restore(restore_path)
    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.

    # results = []
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
        # tqdm.write("global timestep: {}".format(algo.get_policy("default_policy").global_timestep))
        # policy_info = result["info"]["learner"]["default_policy"]["learner_stats"]
        sampler_info = result["sampler_results"]
        mean_reward = sampler_info["episode_reward_mean"]
        bar.set_postfix(rew="{:.2f}".format(mean_reward))
        
        if mean_reward > max_reward or i == iterations - 2:
            max_reward = mean_reward
            if os.name == 'nt':
                tqdm.write("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
            else:
                print("NEW RECORD REACHED. mean_reward: {}".format(max_reward))
                sys.stdout.flush()
            save_ckpt(algo, "experiment_{}_{}".format(exp_name, timestamp), num_ckpt=5, config=config)
            # algo.save("models/experiment_{}_{}".format(exp_name, timestamp))
        bar.update(1)
        # results.append(result)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()

    # pkl.dump(results, open("models\\experiment_{}\\results.dat".format(timestamp), "wb"))


if __name__ == "__main__":
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

    ###############################################################################
    # train MVDQN_ptv2 params
    ###############################################################################
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
            "robot_reward_scale": 2.5,
            "entropy_reward_scale": 2,
            "wrap_obs": False, 
            "use_gymnasium": True
        },
        "num_workers": 6,
        "num_gpus": 1,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": CDPModel_DQN_with_pretrain,
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },

        "min_sample_timesteps_per_iteration": 1000,
        "replay_buffer_config":{
          "type": "MultiAgentReplayBuffer",
          "capacity": 200000,
        },

        # "entropy_coeff": 0.05,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 1000 * 1500,
        },
        "train_batch_size": 64,
        "lr": 5e-4,
        "lr_schedule": [[0, 5e-4], [1000*1000, 5e-4], [1000*1800, 2.5e-4], [1000*2500, 1.25e-4]]
    }
    
    #######################
    # train(
    #     "MVDQN_ptv4_full_LEN300_c3_r", 
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    #######################
    
    ########################
    # ENT ablation study
    # train(
    #     "MVDQN_ptv2_woent_dm_LEN100_retrain", 
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=10000,
    #     restore_path=None
    # )
    ########################

    ########################
    # PT ablation study
    # train(
    #     "MVDQN_woptv2_ent_dm_LEN100_retrain", 
    #     weight_path=None,
    #     config=config,
    #     iterations=10000,
    #     restore_path=None
    # )
    ########################
    
    # train("MVTDQN_ptv3_ent_dm_LEN300", weight_path=os.path.join(weight_root, "state_dict.pt"),
    #       config=config,
    #       iterations=10000,
    #       restore_path=None)
    # evaluate(algo_cls=DQN, restore_path="models/experiment_MVDQN_ptv2_org_LEN50_retrain_1678885630/saved_ckpt", config=config)
    
    
    ###############################################################################
    # train A2C Baseline
    ###############################################################################
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
            "max_episode_length": 100,
            "robot_reward_scale": 3.5,
            "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True
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
    # train_a2c(
    #     "MVA2C_ptv4_full_LEN100_c3_r", 
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2500,
    #     restore_path=None
    # )
    # evaluate(
    #     algo_cls=A2C, 
    #     restore_path="models/experiment_MVA2C_ptv4_full_LEN100_c3_r_1681540711/checkpoint_000926", 
    #     config=config,
    #     recurrent=False
    # )
    ###############################################################################
    # train SAC Baseline
    ###############################################################################
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 6,
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

    # train_sac(
    #     "MVSAC_ptv4_full_LEN100_c3_r", 
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2500,
    #     restore_path=None
    # )

    # evaluate(
    #     algo_cls=SAC, 
    #     restore_path="models/experiment_MVSAC_ptv4_full_LEN100_c3_r_1681354510/checkpoint_001056", 
    #     config=config,
    #     recurrent=False
    # )

    ###############################################################################
    # train alpha zero params
    ###############################################################################
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        # "observation_space": OBSERVATION_SPACE_LITE_DICT, 
        # "action_space": ACTION_SPACE,
        "num_workers": 1,
        "num_gpus": 1,
        
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "custom_model": "cdp_az",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        
        "replay_buffer_config": {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 1000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, episodes or timesteps.
            "storage_unit": "fragments",
        },
        
        "mcts_config": {
           "puct_coefficient": 1.5,
           "num_simulations": 20,
           "temperature": 1.0,
           "dirichlet_epsilon": 0.20,
           "dirichlet_noise": 0.03,
           "argmax_tree_policy": False,
           "add_dirichlet_noise": True,
        },

        "ranked_rewards": {
           "enable": True
        },
        
        "rollout_fragment_length": 10,
        "train_batch_size": 128,
        "sgd_minibatch_size": 32,
        "num_sgd_iter": 1,
        "lr": 2.5e-4,


        # "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    # train_alpha_zero(
    #     "MVAZ_ptv4_full_RAN100_c3_spa", 
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     iterations=2000, 
    #     config=config
    # )
    ###############################################################################
    # evaluate(algo_cls=AlphaZero, restore_path="models/experiment_MVAZ_spa_LEN100_scratch_1680019880/checkpoint_002425", config=config)



    ###############################################################################
    # train MVR2D2_ptv4
    ###############################################################################
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
    # TRAIN MVR2D2_ptv4_full_RAN300_c3_r_reprod
    #####################################
    env_config = {
            "max_episode_length": 300,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 1,
                "reg_scale": 0.05
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN300_c3_r_0.05reg",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN300_c3_r_reprod_1681960940/checkpoint_001028", 
    #          config=config,
    #          recurrent=True)

    #####################################
    # TRAIN MVR2D2_ptv4_full_FIX300_c3_r
    #####################################
    env_config = {
            "max_episode_length": 300,
            "wrap_obs": False, 
            "use_gymnasium": False,
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
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN300_c3_r_reprod_1681960940/checkpoint_001028", 
    #          config=config,
    #          recurrent=True)


    #####################################
    # TRAIN MVR2D2_ptv4_full_RAN300_c4_r
    #####################################
    env_config = {
            # "num_cars": 4, 
            # "num_couriers": 1,
            "max_episode_length": 300,
            "robot_reward_scale": 2,
            "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 1,
                "reg_scale": 0.05
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN300_c4_r_0.05reg",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN300_c4_r_0.05reg_1681975924/checkpoint_001072", 
    #          config=config,
    #          recurrent=True)

    #####################################
    # TRAIN MVR2D2_ptv4_full_RAN300_c5_r
    #####################################
    env_config = {
            # "num_cars": 5, 
            # "num_couriers": 1,
            "max_episode_length": 300,
            "robot_reward_scale": 2,
            "entropy_reward_scale": 2.5,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 1,
                "reg_scale": 0.01
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN300_c5_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN300_c5_r_1682425003/checkpoint_000538", 
    #          config=config,
    #          recurrent=True)


    
    #####################################
    # TRAIN MVR2D2_ptv4_full_RAN200_c3_r
    #####################################
    env_config = {
            "max_episode_length": 200,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 0.5,
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN200_c3_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN200_c3_r_1682426167/checkpoint_001143", 
    #          config=config,
    #          recurrent=True)

    #####################################
    # TRAIN MVR2D2_ptv4_full_RAN200_c4_r
    #####################################
    env_config = {
            "max_episode_length": 200,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 0.5,
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN200_c4_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN200_c4_r_1682426233/checkpoint_000743", 
    #          config=config,
    #          recurrent=True)

    #####################################
    # TRAIN MVR2D2_ptv4_full_RAN200_c5_r
    #####################################
    env_config = {
            "max_episode_length": 200,
            "wrap_obs": False, 
            "use_gymnasium": True,
            "ablation": None,
            "random_episode": True,
            "coeff": {
                "robot_reward_scale": 2,
                "entropy_reward_scale": 2.5,
                "delv_reward_scale": 5,
                "wb_reward_scale": 0.5,
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_RAN200_c5_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN200_c5_r_1682906034/checkpoint_001329", 
    #          config=config,
    #          recurrent=True)
    

    #####################################
    # TRAIN MVR2D2_ptv4_full_LEN300_c3_r
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
                "wb_reward_scale": 0.5,
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_LEN300_c3_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN200_c5_r_1682426415/checkpoint_000595", 
    #          config=config,
    #          recurrent=True)


    #####################################
    # TRAIN MVR2D2_ptv4_full_LEN300_c4_r
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
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_LEN300_c4_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_RAN200_c5_r_1682426415/checkpoint_000595", 
    #          config=config,
    #          recurrent=True)
    
    #####################################
    # TRAIN MVR2D2_ptv4_full_LEN300_c5_r
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
                "reg_scale": 0.0
            }
    }
    config["env_config"] = env_config
    # train_r2d2(
    #     "MVR2D2_ptv4_full_LEN300_c5_r",
    #     weight_path=os.path.join(weight_root, "state_dict.pt"),
    #     config=config,
    #     iterations=2000,
    #     restore_path=None
    # )
    # evaluate(algo_cls=R2D2,
    #          restore_path="models/experiment_MVR2D2_ptv4_full_LEN300_c5_r_1682512843/checkpoint_001373", 
    #          config=config,
    #          recurrent=True)

    from multi_agent_env import OBSERVATION_SPACE, ACTION_SPACE
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "disable_env_checking": True,
        "env": MockEnvWithGroupedAgents,
        "num_workers": 10,
        "num_gpus": 0,
        # "observation_space": OBSERVATION_SPACE,
        # "action_space": ACTION_SPACE,
        
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "lstm_cell_size": 128,
            "max_seq_len": 999999,
        },
        # "replay_buffer_config": {
        #     "type": "ReplayBuffer",
        #     # Size of the replay buffer in batches (not timesteps!).
        #     "capacity": 1000,
        #     # Choosing `fragments` here makes it so that the buffer stores entire
        #     # batches, instead of sequences, episodes or timesteps.
        #     "storage_unit": "fragments",
        # },
        "rollout_fragment_length": 10,
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            # Timesteps over which to anneal epsilon.
            "epsilon_timesteps": 1800 * 4500,

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        "train_batch_size": 256,
        
        "lr": 2.5e-4,

        # "policies":{
        #             "pol1": PolicySpec(
        #                 observation_space=obs_space,
        #                 action_space=act_space,
        #                 config=config.overrides(agent_id=0),
        #             ),
        #             "pol2": PolicySpec(
        #                 observation_space=obs_space,
        #                 action_space=act_space,
        #                 config=config.overrides(agent_id=1),
        #             ),
        #         },
        # "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "pol2" if agent_id else "pol1"
        # "lr_schedule": [[0, 2.5e-4], [1000, 2.5e-4], [1500, 1.25e-4], [2000, 1.25e-4], [3000, 6.25e-5], [4000, 3.125e-5]]
    }
    # train_qmix_test("QMIX_TEST",
    #                 # restore_path="models/experiment_QMIX_TEST_1680257856/checkpoint_001499",
    #                 iterations=5000,
    #                 config=config)
    
    
    
    ray.shutdown()
