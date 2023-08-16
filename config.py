import pickle as pkl
import os

if os.name == 'nt':
    # folder = os.path.split(os.path.abspath(__file__))[0]
    # root, cs_root = os.path.split(folder)
    root = "D:\\liangxingyuan3\\Autonomous Delivery\\数据"
    cs_root = "常熟东南订单"
else:
    root = "/home/liangxingyuan"
    cs_root = "mvac"

courier_root = "快递员数据"
addr_root = "五级地址数据"
user_root = "快递员数据\\小文件"
time_format = "%Y-%m-%d %H:%M:%S.%f"
num_hist = 7

if os.name == 'nt':
    new_data_root = "D:\\liangxingyuan3\\Autonomous Delivery\\multi-view-actor-critic\\data"
else:
    new_data_root = "./data"

with open(os.path.join(new_data_root, "user_list_v2.pkl"), "rb") as f:
    user_list = pkl.load(f)

INDEX_USER = dict([(i+1, u) for i, u in enumerate(user_list)])
USER_INDEX = dict([(u, i+1) for i, u in enumerate(user_list)])

NUM_CARS = 2
NUM_COURIERS = 1

MAX_LEN = 100
NUM_ACTIONS = NUM_CARS * 7 + NUM_COURIERS
NUM_ITERS = 4000
CAR_SPEED = 2.3
MAX_PARCEL_NUM = 35
USE_ALPHAZERO_OBS = False
USE_GYMNASIUM = False
ABLATION = None
R_REWARD_SCALE = 2
E_REWARD_SCALE = 2.5
