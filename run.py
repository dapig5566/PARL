from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

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

import json
import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2"
import time
import sys

from tqdm import tqdm

ModelCatalog.register_custom_model("cdp_model", CDPModel)
ModelCatalog.register_custom_model("cdp_model_dqnver", CDPModel_DQNver)
ModelCatalog.register_custom_model("cdp_qmodel", CDP_QModel)
ModelCatalog.register_custom_model("cdp_sac", CDPSACTorchModel)
ModelCatalog.register_custom_model("cdp_dqn_pt", CDPModel_DQN_with_pretrain)
ModelCatalog.register_custom_model("cdp_az", CDPModel_AlphaZero)
ModelCatalog.register_custom_model("WrappedRecurrentDQN", WrappedRecurrentDQN)
ModelCatalog.register_custom_model("DTWrappedRecurrentTDQN", DTWrappedRecurrentTDQN)

register_env("OrderEnv-v2", OrderEnv2)
register_env("OrderEnv-v3", OrderEnv3)


def save_ckpt(algo, exp_name, num_ckpt=5, config=None):
    algo.save("models/{}".format(exp_name))

    if config is not None:
        if not os.path.exists(os.path.join("models", exp_name, "params.json")):
            try:
                with open(os.path.join("models", exp_name, "params.json"), "w") as f:
                    json.dump(config, f)
            
            except:
                with open(os.path.join("models", exp_name, "params.json"), "w") as f:
                    f.write(str(config))

    ckpt_path = os.path.join(".", "models", exp_name)
    ckpt_files = sorted(os.listdir(ckpt_path))
    if len(ckpt_files) > 5:
        for file in ckpt_files[:-num_ckpt]:
            os.system("rm -r {}".format(os.path.join(ckpt_path, file)))

    

def train_dqn(exp_name, weight_path=None, restore_path=None, iterations=2000, config=None):
    default_config = {
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 8,
        "framework": "torch",
        "model": {
            "custom_model": "cdp_dqn_pt",
            "custom_model_config": {
            },
        },
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
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
  

    algo = DQN(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)


    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
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
           
        bar.update(1)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()



def train_alpha_zero(exp_name, weight_path=None, restore_path=None, iterations=2000, config=None):
    default_config = {
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_rollout_workers": 2,
        "framework": "torch",
        "model": {
            "custom_model": "cdp_az",
            "custom_model_config": {
            },
        },
        "replay_buffer_config": {
            "type": "ReplayBuffer",
            "capacity": 1000,
            "storage_unit": "fragments",
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "lr": 5e-5,
    }
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")

    algo = AlphaZero(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)

    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
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
        bar.update(1)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()


def train_a2c(exp_name, weight_path=None, restore_path=None, iterations=2000, config=None):
    default_config = {
        "lr": 2.5e-5,
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 20,
        "framework": "torch",
        "train_batch_size": 256,
        "model": {
            "custom_model": "cdp_model",
            "custom_model_config": {
            },
        },
        "entropy_coeff": 0.05,
    }
    if config is None:
        config = default_config
    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    algo = A2C(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)

    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
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

            
        bar.update(1)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()


def train_sac(exp_name, weight_path=None, restore_path=None, iterations=2000, config=None):
    default_config = {
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 20,
        "framework": "torch",
        "model": {
            "custom_model": "cdp_sac",
            "custom_model_config": {
            },
        },
        "q_model_config": {
            "custom_model": "cdp_qmodel",
            "custom_model_config": {
            },
        },
        "policy_model_config": {
            "custom_model": "cdp_model",
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
    if config is None:
        config = default_config

    timestamp = int(time.time())
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    log_file = open("logs/log_{}_{}.txt".format(exp_name, timestamp), "w")
    algo = SAC(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        algo.get_policy().model.load_state_dict(state_dict, strict=False)
    if restore_path is not None:
        algo.restore(restore_path)

    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
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
        
        bar.update(1)
        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()



def evaluate(algo_cls, restore_path, config, recurrent=False, env=None):
    from ray.rllib.policy.sample_batch import SampleBatch
    from tabulate import tabulate
    raw_config = {
        "lr": 2.5e-5,
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 2,
        "framework": "torch",
        "model": {
            "custom_model": "cdp_model",
            "custom_model_config": {
            },
        },
        "entropy_coeff": 0.05,
    }
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")
    
    algo = algo_cls(config=config)
  
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

            ob, r, d, _, _ = env.step(action)
            episode_return += r

        results.append(episode_return)
        statistics.append(
            [*env.car_dispatched_num, 
             *env.car_assigned_num, 
             env.courier_dispatched_num, 
             env.courier_assigned_num])
        maps.append(env.dispatch_map)
        routes.append(env.car_routes)

    mean_stat = np.mean(statistics, axis=0)
    
    print()
    print("Evaluation Results:")
        
    table = [
        ["Robot dispatch rate", sum(mean_stat[0:env.num_cars]) / sum(mean_stat[env.num_cars:2*env.num_cars])],
        ["Robot dispatch ratio", sum(mean_stat[env.num_cars: 2*env.num_cars])/mean_stat[-1]],
        ["Robot workload Mean", np.mean(mean_stat[env.num_cars: 2*env.num_cars])],
        ["Robot workload Std", np.std(mean_stat[env.num_cars: 2*env.num_cars])],
        ["Robot delivery entropy Mean", np.mean(env.disp_car_ents)],
        ["Robot delivery entropy Std", np.std(env.disp_car_ents)],
        ["Courier delivery entropy Mean", np.mean(env.disp_courier_ents)],
        ["Courier delivery entropy Std", np.std(env.disp_courier_ents)]
    ]
    headers = ["Metric", "Result"]
    print(tabulate(table, headers=headers))
    print()
    
    return maps, routes



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
            lvl_loss = xent_loss_fn(lvl_pred, y2)
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
            lvl_loss = xent_loss_fn(lvl_pred, eval_y2)

            loss = cls_loss + lvl_loss

            acc = torch.mean(torch.eq(torch.argmax(logits, dim=-1), torch.argmax(eval_y1, dim=-1)).float()).detach().item()
            lvl_acc = torch.mean(torch.eq(torch.argmax(lvl_pred, dim=-1), torch.argmax(eval_y2, dim=-1)).float()).detach().item()

            avg_eval_loss = avg_eval_loss + loss
            avg_eval_acc = avg_eval_acc + acc
            avg_eval_lvl_acc = avg_eval_lvl_acc + lvl_acc
          
        print("EP{} (VAL): loss: {:.5f}, acc: {:.5f}, lvl_acc: {:.5f}".format(i, avg_eval_loss/len(eval_data_loader), avg_eval_acc/len(eval_data_loader), avg_eval_lvl_acc/len(eval_data_loader)))
        if avg_eval_acc/len(eval_data_loader) > max_acc:
            torch.save(model.state_dict(), os.path.join(dir_path, file_name))
            print("New Model Saved.")
            max_acc = avg_eval_acc/len(eval_data_loader)
    
    torch.save(model.state_dict(), os.path.join(dir_path, file_name))


def train_parl(exp_name, weight_path=None, restore_path=None, iterations=2000, config=None):
    default_config = {
        "disable_env_checking": True,
        "env": "OrderEnv-v2",
        "num_workers": 8,
        "framework": "torch",
        "model": {
            "custom_model": "cdp_dqn_pt",
            
            "custom_model_config": {
            },
        },
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
    log_file = open(f"logs/{timestamp}_{exp_name}.log", "w")

    algo = R2D2(config=config)
    if weight_path is not None:
        state_dict = torch.load(weight_path)
        policy_model = algo.get_policy().model
        if isinstance(policy_model, DTWrappedRecurrentTDQN):
            algo.get_policy().model.load_state_dict(state_dict, strict=False)
        elif isinstance(policy_model, WrappedRecurrentDQN):
            algo.get_policy().model.user_embedding.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"policy_model must be DTWrappedRecurrentTDQN or WrappedRecurrentDQN, got {algo.get_policy().model.__class__}")
    
    if restore_path is not None:
        algo.restore(restore_path)
   
    max_reward = -np.inf
    bar = tqdm(range(iterations))
    for i in range(iterations):
        result = algo.train()
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
            save_ckpt(algo, f"{timestamp}_{exp_name}", num_ckpt=5, config=config)
           
        bar.update(1)

        jstr = str(mean_reward)
        log_file.write(jstr)
        log_file.write("\n")
        log_file.flush()

    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    import ray
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--config_path", type=str, default="config/parl.json")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default="./weights/pretrain/TDQN_new_datadist_v4")
    parser.add_argument("--exp_name", type=str, default="PARL_ptv4_rand_reprod")
    args = parser.parse_args()
    ray.init(num_cpus=8)
    
    weight_path = args.weight_path
    ckpt_path = args.ckpt_path
    exp_name = args.exp_name
    num_steps = args.num_steps
    config_path = args.config_path

    with open(config_path, "r") as f:
        config = json.load(f)
    
    if args.do_train:
        train_parl(
            exp_name=exp_name,
            weight_path=os.path.join(weight_path, "state_dict.pt"),
            config=config,
            iterations=num_steps,
            restore_path=ckpt_path
        )
    elif args.do_eval:
        if ckpt_path is None:
            raise RuntimeError("Please set a check point path (--ckpt_path <path>) to do evaluation.")
        evaluate(
            algo_cls=R2D2,
            restore_path=ckpt_path, 
            config=config,
            recurrent=True
        )
    else:
        raise RuntimeError("Please set --do_train or --do_eval to train or evaluate the model.")
    ray.shutdown()
