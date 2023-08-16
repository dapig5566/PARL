from main import R2D2, evaluate
from rdqn_v3 import WrappedRecurrentDQN
from env_v3 import OrderEnv3
import os
import json

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import ray
    ray.init(num_cpus=1)
    with open("config/par2d2.conf", "r") as f:
        all_configs = json.load(f)
    
    config = all_configs["default_config"]
    env_config = all_configs["env_config"]

    config["env_config"] = env_config
    config["num_workers"] = 0

    env = OrderEnv3(config["env_config"])
    evaluate(algo_cls=R2D2,
             restore_path="models/experiment_PAR2D2_v3_c3_1691571004/checkpoint_001822", 
             config=config,
             recurrent=True,
             env=env)
    ray.shutdown()
    