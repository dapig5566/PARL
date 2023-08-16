from gym.envs.registration import EnvSpec
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import datetime
import functools as ft
from config import *
from tabulate import tabulate
import copy

CURRENT_STATE_SPACE = spaces.Tuple(
    [
        spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_LEN, 29 + NUM_ACTIONS)),
        spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_LEN,)),
    ]
)

ORDER_SPACE = spaces.Tuple(
    [
        spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
        spaces.Box(low=-np.inf, high=np.inf, shape=(num_hist, 36)),
        spaces.Box(low=-np.inf, high=np.inf, shape=())
    ]
)

REMAIN_ORDERS_SPACE = spaces.Tuple(
    [
        spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_LEN, 29)),
        spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_LEN,))
    ]
)

DISPATCH_MAP = spaces.Box(low=-np.inf, high=np.inf, shape=(7 * NUM_CARS + NUM_COURIERS, 10, 10))

OBSERVATION_SPACE = spaces.Tuple(
    [
        CURRENT_STATE_SPACE,
        ORDER_SPACE,
        REMAIN_ORDERS_SPACE,
        DISPATCH_MAP
    ]
)

OBSERVATION_SPACE_LITE = spaces.Tuple(
    [
        ORDER_SPACE,
        DISPATCH_MAP
    ]
)

OBSERVATION_SPACE_ABLATION = spaces.Tuple(
    [
        ORDER_SPACE,
        # DISPATCH_MAP
    ]
)

OBSERVATION_SPACE_LITE_DICT = spaces.Dict(
    {
        "obs": OBSERVATION_SPACE_LITE,
        "action_mask": spaces.Box(low=0., high=1., shape=(NUM_ACTIONS,))
    }
)

ACTION_SPACE = spaces.Discrete(NUM_ACTIONS)

def create_dispatch_area(point, order_id):
    return (point[0], point[1]), [order_id]


def find_area(time_segment, point):
    if len(time_segment) == 0:
        return None
    for p, slot in time_segment:
        if p[0] == point[0] and p[1] == point[1]:
            return slot
    return None


def get_or_create_area(time_segment, point, order_id):
    slot = find_area(time_segment, point)
    if slot is not None:
        slot.append(order_id)
    else:
        time_segment.append(create_dispatch_area(point, order_id))


def sort_slot(time_segment):
    lens = []
    for p, slot in time_segment:
        lens.append(len(slot))
    sorted_idx = np.argsort(lens)[::-1]
    return sorted_idx


def find_most_orders(time_segment):
    sorted_idx = sort_slot(time_segment)
    p, slot = time_segment[sorted_idx[0]]
    return p, slot


def find_next_most_orders(time_segments, t, end_t):
    for i in range(t + 1, end_t):
        if len(time_segments[i]) > 0:
            return find_most_orders(time_segments[i])
    return None


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 0.6


def time_used(p1, p2):
    return dist(p1, p2) / CAR_SPEED


class OrderEnv2(gym.Env):
    def __init__(
            self, *args, 
            num_cars=NUM_CARS, 
            num_couriers=NUM_COURIERS, 
            max_episode_length = MAX_LEN,
            wrap_obs=USE_ALPHAZERO_OBS, 
            use_gymnasium=USE_GYMNASIUM,
            ablation=ABLATION,
            random_episode=False
        ):

        coeff = args[0].pop(
            "coeff", 
            {
                "robot_reward_scale": R_REWARD_SCALE,
                "entropy_reward_scale": E_REWARD_SCALE,
                "delv_reward_scale": 5,
                "wb_reward_scale": 0.5,
                "reg_scale": 0.01
            }
        )
        num_cars = args[0].pop("num_cars", num_cars)
        num_couriers = args[0].pop("num_couriers", num_couriers)
        
        robot_reward_scale = coeff.pop("robot_reward_scale", 0)
        entropy_reward_scale = coeff.pop("entropy_reward_scale", 0)
        delv_reward_scale = coeff.pop("delv_reward_scale", 0)
        wb_reward_scale = coeff.pop("wb_reward_scale", 0)
        reg_scale = coeff.pop("reg_scale", 0)

        max_episode_length = args[0].pop("max_episode_length", max_episode_length)
        wrap_obs = args[0].pop("wrap_obs", wrap_obs)
        use_gymnasium = args[0].pop("use_gymnasium", use_gymnasium)
        ablation = args[0].pop("ablation", ablation)
        random_episode = args[0].pop("random_episode", random_episode)
        color_before = "\033[0;36;40m"
        color_after = "\033[0m"
        # print("\033[0;36;40mEnvironment Params:\033[0m")
        print("\033[7;36;40mEnvironment Params:\033[0m")
        
        table = [["num_cars", num_cars],
                 ["num_couriers", num_couriers],
                 ["robot_reward_scale", robot_reward_scale],
                 ["entropy_reward_scale", entropy_reward_scale],
                 ["max_episode_length", max_episode_length],
                 ["wrap_obs", wrap_obs],
                 ["use_gymnasium", use_gymnasium],
                 ["ablation", ablation],
                 ["random_episode", random_episode],
                 ["delv_reward_scale", delv_reward_scale],
                 ["wb_reward_scale", wb_reward_scale],
                 ["reg_scale", reg_scale],]
        headers = ["Param", "Value"]
        for row in table:
            row[0] = color_before + row[0] + color_after
        for i, attr in enumerate(headers):
            headers[i] = color_before + headers[i] + color_after
        print(tabulate(table, headers=headers))
        print()

        self.delv_reward_scale = delv_reward_scale
        self.wb_reward_scale = wb_reward_scale
        self.reg_scale = reg_scale
        self.random_episode = random_episode
        self.wrap_obs = wrap_obs
        self.courier_assigned_num = 0
        self.car_assigned_num = [0 for _ in range(NUM_CARS)]
        self.courier_dispatched_num = 0
        self.car_dispatched_num = [0 for _ in range(NUM_CARS)]
        self.dispatch_time_error = []

        self.ablation = ablation
        self.action_space = ACTION_SPACE
        
        self.observation_space = OBSERVATION_SPACE_LITE_DICT if self.wrap_obs else OBSERVATION_SPACE_LITE
        self.spec = EnvSpec("OrderEnv-v2")

        self.steps = 0
        self.max_episode_length = max_episode_length
        self.robot_reward_scale = robot_reward_scale
        self.entropy_reward_scale = entropy_reward_scale
        self.use_gymnasium = use_gymnasium
        self.num_cars = num_cars
        self.num_couriers = num_couriers

        data_files = [
            "dataset2train_v2.pkl",
            "userdata2train_v2.pkl"
        ]
        exists = [os.path.exists(os.path.join(new_data_root, path)) for path in data_files]
        if all(exists):
            with open(os.path.join(new_data_root, data_files[0]), "rb") as f:
                self.data = pkl.load(f)
            with open(os.path.join(new_data_root, data_files[1]), "rb") as f:
                self.user_data = pkl.load(f)
        else:
            with open(os.path.join(new_data_root, "prepared_dataset_v2.pkl"), "rb") as f:
                self.data = pkl.load(f)
            with open(os.path.join(new_data_root, "prepared_user_data_v2.pkl"), "rb") as f:
                self.user_data = pkl.load(f)
            self.preprocess()
            with open(os.path.join(new_data_root, data_files[0]), "wb") as f:
                pkl.dump(self.data, f)
            with open(os.path.join(new_data_root, data_files[1]), "wb") as f:
                pkl.dump(self.user_data, f)

        self.next_users, self.next_orders, self.next_labels, self.next_dates, self.next_succdelv_td, self.next_is_uav, self.next_is_redelv, self.next_levels = self.sample_day()
        self.spec.max_episode_steps = min(self.max_episode_length, len(self.next_users))
        # self.max_episode_steps = min(MAX_LEN, len(self.next_users))
        self.dispatch_state = []
        dispatch_map = [np.zeros([7, 10, 10])] * NUM_CARS + [np.zeros([1, 10, 10])] * NUM_COURIERS
        self.dispatch_map = np.concatenate(dispatch_map)
        saved_actions = {}
        saved_actions.update({
            "car_{}".format(i): [[] for _ in range(7)] for i in range(NUM_CARS)
        })
        saved_actions.update({
            "courier": []
        })
        self.saved_actions = saved_actions
        self.num_lv5 = 0
        self.num_lv4 = 0
        self.num_lv3 = 0
        self.disp_order_level_car = [[] for _ in range(NUM_CARS)]
        self.disp_order_level_courier = []
        self.abandoned_orders = []
        self.car_routes = {
            "car_{}".format(i): [[] for _ in range(7)] for i in range(NUM_CARS)
        }
        self.disp_car_ents = []
        self.disp_courier_ents = []

    def reset(self, **kwargs):
        self.disp_car_ents = []
        self.disp_courier_ents = []

        self.courier_assigned_num = 0
        self.car_assigned_num = [0 for _ in range(NUM_CARS)]
        self.courier_dispatched_num = 0
        self.car_dispatched_num = [0 for _ in range(NUM_CARS)]
        self.dispatch_time_error = []

        self.steps = 0
        self.dispatch_state = []
        self.next_users, self.next_orders, self.next_labels, self.next_dates, self.next_succdelv_td, self.next_is_uav, \
            self.next_is_redelv, self.next_levels = self.sample_day()
        self.spec.max_episode_steps = min(self.max_episode_length, len(self.next_users))
        # self.max_episode_steps = min(MAX_LEN, len(self.next_users))
        dispatch_map = [np.zeros([7, 10, 10])] * NUM_CARS + [np.zeros([1, 10, 10])] * NUM_COURIERS
        self.dispatch_map = np.concatenate(dispatch_map)
        saved_actions = {}
        saved_actions.update({
            "car_{}".format(i): [[] for _ in range(7)] for i in range(NUM_CARS)
        })
        saved_actions.update({
            "courier": []
        })
        self.saved_actions = saved_actions
        self.disp_order_level_car = [[] for _ in range(NUM_CARS)]
        self.disp_order_level_courier = []
        self.abandoned_orders = []
        self.car_routes = {
            "car_{}".format(i): [[] for _ in range(7)] for i in range(NUM_CARS)
        }
        if self.use_gymnasium:
            return self.get_observation(), {}
        else:
            return self.get_observation()
    
    def get_state(self):
        state = {
            "courier_assigned_num": self.courier_assigned_num,
            "car_assigned_num": self.car_assigned_num,
            "courier_dispatched_num": self.courier_dispatched_num,
            "car_dispatched_num": self.car_dispatched_num,
            "dispatch_time_error": self.dispatch_time_error,
            "steps": self.steps,
            "dispatch_state": self.dispatch_state,
            "episode": self.episode,
            "max_episode_steps": self.spec.max_episode_steps,
            # "max_episode_steps": self.max_episode_steps,
            "dispatch_map": self.dispatch_map,
            "saved_actions": self.saved_actions,
            "disp_order_level_car": self.disp_order_level_car,
            "disp_order_level_courier": self.disp_order_level_courier,
            "abandoned_orders": self.abandoned_orders,
            "car_routes": self.car_routes,
        }
        return copy.deepcopy(state)
    
    def set_state(self, state):
        self.courier_assigned_num = state["courier_assigned_num"]
        self.car_assigned_num = state["car_assigned_num"]
        self.courier_dispatched_num = state["courier_dispatched_num"]
        self.car_dispatched_num = state["car_dispatched_num"]
        self.dispatch_time_error = state["dispatch_time_error"]
        self.steps = state["steps"]
        self.dispatch_state = state["dispatch_state"]
        self.episode = state["episode"]
        self.next_users, self.next_orders, self.next_labels, self.next_dates, self.next_succdelv_td, \
            self.next_is_uav, self.next_is_redelv, self.next_levels = self.sample_day(episode=self.episode)
        self.spec.max_episode_steps = state["max_episode_steps"]
        # self.max_episode_steps = state["max_episode_steps"]
        self.dispatch_map = state["dispatch_map"]
        self.saved_actions = state["saved_actions"]
        self.disp_order_level_car = state["disp_order_level_car"]
        self.disp_order_level_courier = state["disp_order_level_courier"]
        self.abandoned_orders = state["abandoned_orders"]
        self.car_routes = state["car_routes"]
        
        return self.get_observation()


    def get_remained_orders(self):
        remained_uids = [USER_INDEX[u] for u in self.next_users[self.steps + 1:self.spec.max_episode_steps]]
        remained_orders = list(self.next_orders[self.steps + 1:self.spec.max_episode_steps])
        current_len = len(remained_uids)
        if self.spec.max_episode_steps > current_len:
            remained_uids = remained_uids + [0 for _ in range(self.spec.max_episode_steps - current_len)]
            remained_orders = remained_orders + [np.zeros([29]) for _ in range(self.spec.max_episode_steps - current_len)]
        remained_uids = np.array(remained_uids)
        remained_orders = np.array(remained_orders)
        # return np.ones([MAX_LEN, 29]), np.zeros([MAX_LEN])
        return remained_orders, remained_uids

    def get_observation(self):
        if self.steps >= self.spec.max_episode_steps:
        # if self.steps >= self.max_episode_steps:
            # return self.process_state(self.dispatch_state), \
            #        (np.zeros([29]), np.zeros([num_hist, 36]), 0), \
            #        self.get_remained_orders(), \
            #        self.dispatch_map
            
            # return (np.zeros([29]), np.zeros([num_hist, 36]), 0), self.dispatch_map
            
            if self.wrap_obs:
                return {"obs":((np.zeros([29]), np.zeros([num_hist, 36]), 0), self.dispatch_map),
                        "action_mask": np.zeros([NUM_ACTIONS])}
            else:
                return (np.zeros([29]), np.zeros([num_hist, 36]), 0), self.dispatch_map
            
            # return ((np.zeros([29]), np.zeros([num_hist, 36]), 0), ) # ablation
        else:
            order_to_dispatch = self.next_orders[self.steps]
            order_date = self.next_dates[self.steps]
            uid = USER_INDEX[self.next_users[self.steps]]
            all_user_hist, all_user_dates, _, _, _ = self.user_data[self.next_users[self.steps]]
            hist_can_access_idx = 0
            for i in range(len(all_user_dates)):
                if all_user_dates[i] < order_date:
                    hist_can_access_idx = i
                    break
            
            user_hist = all_user_hist[hist_can_access_idx:]
            user_hist = [h for h in user_hist
                         if np.argmax(h[7:7+10]).item() == np.argmax(order_to_dispatch[7:7+10]).item() \
                         and np.argmax(h[17:17+10]).item() == np.argmax(order_to_dispatch[17:17+10]).item()][:num_hist]
            
            user_hist = np.array(user_hist)

            assert len(user_hist) <= num_hist, ValueError(f"user_hist need to be less than {num_hist}, got {len(user_hist)}.")
            if len(user_hist) == 0:
                user_hist = np.zeros([num_hist, 36])
            elif len(user_hist) < num_hist:
                user_hist = np.concatenate([user_hist, np.zeros([num_hist - len(user_hist), 36])])
            
            # return self.process_state(self.dispatch_state), \
            #        (order_to_dispatch, user_hist, uid), \
            #        self.get_remained_orders(), \
            #        self.dispatch_map
            
            # return (order_to_dispatch, user_hist, uid), self.dispatch_map
            if self.wrap_obs:
                return {"obs":((order_to_dispatch, user_hist, uid), self.dispatch_map / 20),
                        "action_mask": np.ones([NUM_ACTIONS])}
            else:
                return (order_to_dispatch, user_hist, uid), self.dispatch_map / 20
            # return ((order_to_dispatch, user_hist, uid), ) # ablation

        # return (np.ones([29]), np.ones([num_hist, 36]), 0)

    def process_state(self, dispatch_state):
        current_len = len(dispatch_state)
        if self.spec.max_episode_steps > current_len:
            padded_dispatch_state = [i for i in dispatch_state] + [(0, np.zeros([29 + NUM_ACTIONS]))
                                                                   for _ in range(self.spec.max_episode_steps - current_len)]
        else:
            padded_dispatch_state = dispatch_state
        uids, orders = zip(*padded_dispatch_state)
        uids = np.array(uids)
        orders = np.array(orders)
        return orders, uids
        # return np.zeros([MAX_LEN, 29 + NUM_CARS * 7 + NUM_COURIERS]), np.zeros([MAX_LEN])
        # return np.zeros([MAX_LEN, 29 + NUM_CARS * 7 + NUM_COURIERS])

    
    def step(self, action):
        self.dispatch_state.append(
            (
                USER_INDEX[self.next_users[self.steps]],
                np.concatenate([self.next_orders[self.steps], np.eye(self.action_space.n)[action]], axis=-1)
            )
        )
        x = int(np.argmax(self.next_orders[self.steps][7:7 + 10]))
        y = int(np.argmax(self.next_orders[self.steps][7 + 10:7 + 10 + 10]))

        self.dispatch_map[action, y, x] += 1

        is_car = action < 7 * NUM_CARS
        if is_car:
            car_id = int(action // 7)
            t = action % 7
            self.car_assigned_num[car_id] += 1
            get_or_create_area(self.saved_actions["car_{}".format(car_id)][t], (x, y), self.steps)
        else:
            t = None
            self.courier_assigned_num += 1
            get_or_create_area(self.saved_actions["courier"], (x, y), self.steps)

        # if is_car:
        #     r = int(np.argmax(self.next_labels[self.steps]) == t)
        # else:
        #     dist = np.argmax(self.next_labels[self.steps]) - t
        #     r = dist if dist < 0 else min(max(7 - dist, 0), 7) * self.courier_reward_scale
        order_lv, ent = self.next_levels[self.steps]
        if is_car:

            if np.argmax(self.next_labels[self.steps]) == t:
                # if order_lv == 5:
                #     r = 20
                # elif order_lv == 4:
                #     r = 25
                # elif order_lv == 3:
                #     r = 26
                # else:
                #     raise ValueError("Order level not in proper value range.")

                # entrew testing.
                if self.wrap_obs or (self.ablation is not None and self.ablation=="ent"):
                    r = 0
                else:
                    r = ent * self.entropy_reward_scale * self.robot_reward_scale
                # r = 0 # for alpha_zero
                self.car_dispatched_num[car_id] += 1
                self.disp_order_level_car[car_id].append(order_lv)
                self.disp_car_ents.append(ent)
            else:
                # if order_lv == 5:
                #     r = -10
                # elif order_lv == 4:
                #     r = -5
                # elif order_lv == 3:
                #     r = -3
                # else:
                #     raise ValueError("Order level not in proper value range.")

                # entrew testing.
                if self.wrap_obs or (self.ablation is not None and self.ablation=="ent"):
                    r = 0
                else:
                    r = -ent * self.entropy_reward_scale * self.robot_reward_scale
                # r = 0 # for alpha_zero
            # if self.car_assigned_num[car_id] > MAX_PARCEL_NUM:
            #     r = -10

        else:
            # if order_lv == 5:
            #     r = 1.5
            # elif order_lv == 4:
            #     r = 10
            # elif order_lv == 3:
            #     r = 20
            # else:
            #     raise ValueError("Order level not in proper value range.")

            # entrew testing.
            if self.wrap_obs or (self.ablation is not None and self.ablation=="ent"):
                r = 0
            else:
                r = ent * self.entropy_reward_scale
            # r = 0 # for alpha_zero
            # r = 5 * self.courier_reward_scale
            self.courier_dispatched_num += 1
            self.disp_order_level_courier.append(order_lv)
            self.disp_courier_ents.append(ent)
            # r = -5

        self.steps += 1
        if self.steps >= self.spec.max_episode_steps:
        # if self.steps >= self.max_episode_steps:
            r += self.compute_reward()
        # if self.steps >= self.spec.max_episode_steps:
        #     r = self.compute_reward()
        # else:
        #     if is_car:
        #         r = float(np.argmax(self.next_labels[self.steps]) == t) * 0.5
        #     else:
        #         r = 0
        if self.use_gymnasium:
            return self.get_observation(), r, self.steps >= self.spec.max_episode_steps, False, {}
        else:
            return self.get_observation(), r, self.steps >= self.spec.max_episode_steps, {}
        # return self.get_observation(), r, self.steps >= self.max_episode_steps, False, {}

    def dispatch_orders(self):
        def find_nearest(start_pt, areas):
            min_dist = np.inf
            nearest_idx = 0
            for i, a in enumerate(areas):
                if areas_checked[i]:
                    continue
                d = dist(start_pt, a[0])
                if d < min_dist:
                    min_dist = d
                    nearest_idx = i
            return nearest_idx

        coords = {
            "car_{}".format(i): (2, 6) for i in range(NUM_CARS)
        }
        # coords["courier"] = (2, 6)
        remained_areas = {
            "car_{}".format(i): [[] for _ in range(7)] for i in range(NUM_CARS)
        }
        completed_orders = []
        for t in range(0, 3):
            for car_id in range(NUM_CARS):
                areas_to_dispatch = self.saved_actions["car_{}".format(car_id)][t]
                areas_checked = [False] * len(areas_to_dispatch)
                time_remained = 2
                while sum(areas_checked) < len(areas_to_dispatch):
                    nearest_idx = find_nearest(coords["car_{}".format(car_id)], areas_to_dispatch)
                    areas_checked[nearest_idx] = True

                    if time_remained > time_used(coords["car_{}".format(car_id)], areas_to_dispatch[nearest_idx][0]):
                        self.car_routes["car_{}".format(car_id)][t].append(areas_to_dispatch[nearest_idx][0])
                        for order_id in areas_to_dispatch[nearest_idx][1]:
                            if np.argmax(self.next_labels[order_id]) == t:
                                completed_orders.append(order_id)
                        time_remained -= time_used(coords["car_{}".format(car_id)], areas_to_dispatch[nearest_idx][0])
                        coords["car_{}".format(car_id)] = areas_to_dispatch[nearest_idx][0]
                    else:
                        remained_areas["car_{}".format(car_id)][t].append(areas_to_dispatch[nearest_idx])

        coords = {
            "car_{}".format(i): (2, 6) for i in range(NUM_CARS)
        }
        for t in range(3, 7):
            for car_id in range(NUM_CARS):
                areas_to_dispatch = self.saved_actions["car_{}".format(car_id)][t]
                areas_checked = [False] * len(areas_to_dispatch)
                time_remained = 2
                while sum(areas_checked) < len(areas_to_dispatch):
                    nearest_idx = find_nearest(coords["car_{}".format(car_id)], areas_to_dispatch)
                    areas_checked[nearest_idx] = True

                    if time_remained > time_used(coords["car_{}".format(car_id)], areas_to_dispatch[nearest_idx][0]):
                        for order_id in areas_to_dispatch[nearest_idx][1]:
                            if np.argmax(self.next_labels[order_id]) == t:
                                completed_orders.append(order_id)
                        time_remained -= time_used(coords["car_{}".format(car_id)], areas_to_dispatch[nearest_idx][0])
                        coords["car_{}".format(car_id)] = areas_to_dispatch[nearest_idx][0]
                    else:
                        remained_areas["car_{}".format(car_id)][t].append(areas_to_dispatch[nearest_idx])

        return completed_orders, remained_areas

    def compute_reward(self):  # 全1reward，需要entropy来放松假设，解决稀疏反馈大空间搜索问题
        completed_orders, remained_areas = self.dispatch_orders()
        r1 = len(completed_orders)
        remained_orders = []
        for car_id in range(NUM_CARS):
            for t in range(7):
                for a in remained_areas["car_{}".format(car_id)][t]:
                    remained_orders.extend(a[1])

        r2 = len(remained_orders)
        r3 = np.std(self.car_assigned_num)
        
        min_car_order = self.spec.max_episode_steps / 2 / NUM_CARS
        regularity = np.array([min(num, 0) for num in (np.array(self.car_assigned_num) - min_car_order)])
        
        if self.ablation is not None and self.ablation == "ent":
            return self.robot_reward_scale*(r1-r2) + self.courier_dispatched_num + 5 * (r1 - r2 + 0.5*sum(regularity)) - 1 * r3
        else:
            return self.delv_reward_scale * (r1 - r2 + self.reg_scale*sum(regularity)) - self.wb_reward_scale * r3
            # return self.delv_reward_scale * (r1 - r2) - self.wb_reward_scale * r3
            # return 5 * (r1 - r2 + 0.1*sum(regularity)) - 0.5 * r3

    def show_order_dist(self):
        car_assignment_dist = np.zeros([5])
        courier_assignment_dist = np.zeros([5])
        print("Order Distribution:")
        print()
        for i in range(NUM_CARS):
            dist_dict = {}
            for j in self.disp_order_level_car[i]:
                if j in dist_dict:
                    dist_dict[j] += 1
                else:
                    dist_dict[j] = 1
                car_assignment_dist[j-1] = car_assignment_dist[j-1] + 1
            print("car_{} distribution: {}".format(i, dist_dict))
        print()
        dist_dict = {}
        for j in self.disp_order_level_courier:
            if j in dist_dict:
                dist_dict[j] += 1
            else:
                dist_dict[j] = 1
            courier_assignment_dist[j-1] = courier_assignment_dist[j-1] + 1
        print("courier distribution: {}".format(dist_dict))
        print()
        return {"courier": courier_assignment_dist, "car": car_assignment_dist}

    #     courier_reward = 0
    #     for _ in range(NUM_COURIERS):
    #         area_data = self.saved_actions["courier"]
    #         for d in area_data:
    #             courier_reward += len(d[1])
    #             self.courier_dispatched_num += len(d[1])
    #         # dists = []
    #         # for d in area_data:
    #         #     dists.append(dist((2, 6), d[0]))
    #         # sorted_idx = np.argsort(dists)
    #         # current_point = (2, 6)
    #         # current_time = 8
    #         # for i in sorted_idx:
    #         #     current_time += time_used(area_data[i][0], current_point) + 0.6
    #         #     r_time = (int(np.floor(current_time)) - 8) // 2
    #         #     current_point = area_data[i][0]
    #         #     for order_id in area_data[i][1]:
    #         #         d = np.argmax(self.next_labels[order_id]) - r_time
    #         #         r = d if d < 0 else min(max(7 - d, 0), 7)
    #         #         if d > 0:
    #         #             self.courier_dispatched_num += 1
    #         #         self.dispatch_time_error.append(np.abs(d))
    #         #         courier_reward += r * self.courier_reward_scale
    #
    #     return car_reward + courier_reward * 5 * self.courier_reward_scale
    #     # return car_reward + courier_reward

    def sample_day(self, episode=None):
        # month = np.random.choice(len(self.data))
        month = 0

        if episode is not None:
            day = episode
        else:
            day = np.random.choice(len(self.data[month]))
        
        self.episode = day
        day_orders = self.data[month][day]
        
        # random order set
        if self.random_episode:
            indices = list(sorted(np.random.choice(len(day_orders), size=min(self.max_episode_length, len(day_orders)), replace=False)))
            day_orders = [day_orders[i] for i in indices]
        
        next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels = zip(
            *day_orders)

        return next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels

    def __next__(self):
        month_data = self.data[0]
        # for month_data in self.data:
        for day_data in month_data:
            next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels = zip(
                *day_data)
            yield next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels

    def filter_data(self, order_data):
        lv, ent, original_hist_length = self.get_order_level(order_data)

        level_dist = [0.29429714, 0.18390418, 0.03529596, 0.42141304, 0.06508968]
        prob = 0.1
        proportion = (3, 2, 1, 1, 1)

        if original_hist_length < num_hist:
            return False, (lv, ent)

        # proportion = (3, 2, 1)
        
        
        # if np.random.uniform() < prob:
        #     return True, (lv, ent)
        # else:
        #     return False, (lv, ent)


        if (lv >= 4 and lv <=5):
             return True, (lv, ent)
        elif lv == 3:
            return True, (lv, ent)
            # if self.num_lv3 < 150:
            #     self.num_lv3 += 1
            #     return True, (lv, ent)
            # else:
            #     return False, (lv, ent)
        elif lv == 2:
            # return True, (lv, ent)
            if self.num_lv2 < 150:
                self.num_lv2 += 1
                return True, (lv, ent)
            else:
                return False, (lv, ent)
        elif lv == 1:
            if self.num_lv1 < 100:
                self.num_lv1 += 1
                return True, (lv, ent)
            else:
                return False, (lv, ent)
        else:
            return False, (lv, ent)
            
        # if lv == 5:
        #     self.num_lv5 += 1
        #     return True, (lv, ent)
        # elif lv == 4:
        #     if self.num_lv5 > self.num_lv4 * (proportion[0]/proportion[1]):
        #         self.num_lv4 += 1
        #         return True, (lv, ent)
        #     else:
        #         return False, (lv, ent)
        # elif lv == 1:
        #     if self.num_lv4 > self.num_lv1 * (proportion[1]/proportion[2]):
        #         self.num_lv1 += 1
        #         return True, (lv, ent)
        #     else:
        #         return False, (lv, ent)
        # elif lv == 3:
        #     if self.num_lv4 > self.num_lv3 * (proportion[1]/proportion[2]):
        #         self.num_lv3 += 1
        #         return True, (lv, ent)
        #     else:
        #         return False, (lv, ent)
        # elif lv == 2:
        #     if self.num_lv3 > self.num_lv2 * (proportion[2]/proportion[3]):
        #         self.num_lv2 += 1
        #         return True, (lv, ent)
        #     else:
        #         return False, (lv, ent)
        # elif lv == 1:
        #     if self.num_lv2 > self.num_lv1 * (proportion[3]/proportion[4]):
        #         self.num_lv1 += 1
        #         return True, (lv, ent)
        #     else:
        #         return False, (lv, ent)
        # else:
        #     return False, (lv, ent)

    def reset_filter(self):
        self.num_lv5 = 0
        self.num_lv4 = 0
        self.num_lv3 = 0
        self.num_lv2 = 0
        self.num_lv1 = 0

    def get_order_level(self, order_data):
        u, o, l, order_date, _, _, _ = order_data

        order = (
            np.argmax(o[:7]).item(),
            np.argmax(o[7:7 + 10]).item(),
            np.argmax(o[17:17 + 10]).item(),
            np.argmax(l).item()
        )

        all_user_hist, all_user_dates, _, _, _ = self.user_data[u]

        hist_can_access_idx = 0
        for i in range(len(all_user_dates)):
            if all_user_dates[i] < order_date:
                hist_can_access_idx = i
                break

        user_record = [(
            np.argmax(d[:7]).item(),
            np.argmax(d[7:7 + 10]).item(),
            np.argmax(d[17:17 + 10]).item(),
            np.argmax(d[-7:]).item()
        ) for d in all_user_hist[hist_can_access_idx:]]

        day_hist_orders = [o for o in user_record if o[1] == order[1] and o[2] == order[2]]
        day_hist_orders = day_hist_orders[:num_hist]
        original_hist_length = len(day_hist_orders)
        #         day_hist_orders = user_record[:num_hist]

        if len(day_hist_orders) == 0:
            order_distrib = np.ones([7], dtype=np.float32)
        else:
            order_distrib = np.sum([np.eye(7)[o[3]] for o in day_hist_orders], axis=0)

        order_distrib = order_distrib - np.max(order_distrib)
        order_distrib_exp = np.exp(order_distrib)
        order_distrib_softmax = order_distrib_exp / np.sum(order_distrib_exp)
        ent = -np.sum(np.log(order_distrib_softmax + 1e-8) * order_distrib_softmax)

        num_same_order = sum(
            [order[0] == i[0] and order[1] == i[1] and order[2] == i[2] and order[3] == i[3]
             for i in day_hist_orders]
        )
        num_same_area_order = sum(
            [order[1] == i[1] and order[2] == i[2] and order[3] == i[3]
             for i in day_hist_orders]
        )

        # if num_same_order > 1:
        #     easy_lv = 5
        # elif num_same_area_order > 1:
        #     easy_lv = 4
        # elif num_same_order == 1:
        #     easy_lv = 3
        # elif num_same_area_order == 1:
        #     easy_lv = 2
        # else:
        #     easy_lv = 1
        if num_same_order > 5 or num_same_area_order > 5:
            easy_lv = 5
        elif num_same_order > 3 or num_same_area_order > 3:
            easy_lv = 4
        elif num_same_order > 2 or num_same_area_order > 2:
            easy_lv = 3
        elif num_same_order > 1 or num_same_area_order > 1:
            easy_lv = 2
        else:
            easy_lv = 1
        return easy_lv, ent, original_hist_length

    def preprocess(self):
        processed_user_data = {}
        for u in self.user_data.keys():
            orders = []
            for i in self.user_data[u]:
                acct = i[0]
                weekday = i[1] // 10
                assert 1 <= weekday <= 7
                weekday = np.eye(7)[weekday - 1]
                r_time = i[1] % 10
                r_time = min(r_time, 6)
                r_time = np.eye(7)[r_time]
                x = i[2]
                x = np.eye(10)[x]
                y = i[3]
                y = np.eye(10)[y]

                if not np.isnan(float(i[4])) or not np.isnan(float(i[6])):
                    w = float(i[4]) if not np.isnan(float(i[4])) else float(i[6])
                else:
                    continue
                if not np.isnan(float(i[5])) or not np.isnan(float(i[7])):
                    v = float(i[5]) if not np.isnan(float(i[5])) else float(i[7])
                else:
                    continue
                date = i[8]
                succdelv_td = i[9]
                is_uav = i[10]
                is_redelv = i[11]
                orders.append((
                    np.concatenate([weekday, x, y, np.array([w, v]), r_time]),
                    date,
                    succdelv_td,
                    is_uav,
                    is_redelv
                ))

            if len(orders) == 0:
                orders = [(
                    np.concatenate([np.zeros([7]), [0], [0], np.array([1.0, 1.0]), np.zeros([7])]),
                    datetime.date(2021, 7, 1),
                    0,
                    False,
                    False
                ) for _ in range(7)]
            hist_orders, order_date, succdelv_td, is_uav, is_redelv = zip(*orders)
            processed_user_data[u] = (np.array(hist_orders), order_date, succdelv_td, is_uav, is_redelv)

        wvs = [[k[-9], k[-8]] for i in processed_user_data.keys() for k in processed_user_data[i][0]]
        mean_wvs = np.mean(wvs, axis=0)
        std_wvs = np.std(wvs, axis=0)

        for i in processed_user_data.keys():
            for k in processed_user_data[i][0]:
                k[-9:-7] = (k[-9:-7] - mean_wvs) / std_wvs
        self.user_data = processed_user_data

        processed_data = []
        for mon_data in self.data:
            processed_mon_data = []
            for day_data in mon_data:
                processed_day_data = []
                self.reset_filter()
                for ind, i in enumerate(day_data):
                    acct = i[0]
                    weekday = i[1] // 10

                    if not (1 <= weekday <= 7):
                        print(i)
                        assert False
                    weekday = np.eye(7)[weekday - 1]
                    r_time = i[1] % 10
                    r_time = min(r_time, 6)
                    r_time = np.eye(7)[r_time]

                    x = i[2]
                    x = np.eye(10)[x]
                    y = i[3]
                    y = np.eye(10)[y]
                    if not np.isnan(i[4]) or not np.isnan(i[6]):
                        w = i[4] if not np.isnan(i[4]) else i[6]
                    else:
                        continue
                    if not np.isnan(i[5]) or not np.isnan(i[7]):
                        v = i[5] if not np.isnan(i[5]) else i[7]
                    else:
                        continue
                    date = i[8]
                    succdelv_td = i[9]
                    is_uav = i[10]
                    is_redelv = i[11]
                    order_data = (
                        acct,
                        np.concatenate([weekday, x, y, np.array([w, v])]),
                        r_time,
                        date,
                        succdelv_td,
                        is_uav,
                        is_redelv
                    )
                    if acct not in self.user_data:
                        continue
                    # if is_uav:
                    #     is_use = True
                    # else:
                    #     is_use, order_lv = self.filter_data(order_data)
                    is_use, order_lv = self.filter_data(order_data)

                    if is_use:
                        processed_day_data.append((*order_data, order_lv))
                processed_mon_data.append(processed_day_data)
            processed_data.append(processed_mon_data)

        wvs = [[k[1][-2], k[1][-1]] for i in processed_data for j in i for k in j]
        mean_wvs = np.mean(wvs, axis=0)
        std_wvs = np.std(wvs, axis=0)
        for i in processed_data:
            for j in i:
                for k in j:
                    k[1][-2:] = (k[1][-2:] - mean_wvs) / std_wvs
        self.data = processed_data


def guo(e, i):
    u = e.next_users[i]
    o = e.next_orders[i]
    l = e.next_labels[i]

    print("({}, ({}, {}), {})".format(np.argmax(o[:7]).item(),
                                      np.argmax(o[7:7 + 10]).item(),
                                      np.argmax(o[17:17 + 10]).item(),
                                      np.argmax(l).item()))
    print()
    for d in e.user_data[u][:3]:
        print("({}, ({}, {}), {})".format(np.argmax(d[:7]).item(),
                                          np.argmax(d[7:7 + 10]).item(),
                                          np.argmax(d[17:17 + 10]).item(),
                                          np.argmax(d[-7:]).item()))


def get_data_by_order(data, i, num_hist, debug=False):
    next_users, next_orders, next_labels, next_dates = data
    u = next_users[i]
    o = next_orders[i]
    l = next_labels[i]
    order_date = next_dates[i]

    order = (
        np.argmax(o[:7]).item(),
        np.argmax(o[7:7 + 10]).item(),
        np.argmax(o[17:17 + 10]).item(),
        np.argmax(l).item()
    )

    all_user_hist, all_user_dates, _, _, _ = e.user_data[u]

    hist_can_access_idx = 0
    for i in range(len(all_user_dates)):
        if all_user_dates[i] < order_date:
            hist_can_access_idx = i
            break

    user_record = [(
        np.argmax(d[:7]).item(),
        np.argmax(d[7:7 + 10]).item(),
        np.argmax(d[17:17 + 10]).item(),
        np.argmax(d[-7:]).item()
    ) for d in all_user_hist[hist_can_access_idx:]]

    #     day_hist_orders = [h for h in user_record if h[0] == order[0]]
    #     day_hist_orders = day_hist_orders[:num_hist]

    day_hist_orders = [o for o in user_record if o[1] == order[1] and o[2] == order[2]]
    day_hist_orders = day_hist_orders[:num_hist]

    num_in_day = sum([order[0] == i[0] and order[3] == i[3] for i in day_hist_orders])
    num_in_area = sum([order[1] == i[1] and order[2] == i[2] and order[3] == i[3] for i in day_hist_orders])
    num_all_same = sum(
        [order[0] == i[0] and order[1] == i[1] and order[2] == i[2] and order[3] == i[3] for i in day_hist_orders])

    return (order, order_date), day_hist_orders, num_in_day, num_in_area, num_all_same


def count_orders(e, attritube, num_hist, debug=False):
    assert attritube in ["same_day", "same_area", "same_all"]
    attr_dict = {
        "same_day": 2,
        "same_area": 3,
        "same_all": 4
    }
    attritube = attr_dict[attritube]
    has_order_count = 0
    for i in range(50):
        if get_data_by_order(e, i, num_hist)[attritube] > 0:
            has_order_count += 1

    # if debug:
    #     if has_order_count == 0:
    #         for i in range(50):
    #             a, b, _, _, s = get_data_by_order(e, i, num_hist, debug=True)
    #             if s == 0:
    #                 print(e.next_users[i])
    #                 print(a)
    #                 print()
    #                 for i in b:
    #                     print(i)
    #                 print()
    #
    #         assert False

    return has_order_count


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # import seaborn
    # e = OrderEnv()
    # rewards = []
    # for i in range(53):
    #     e.reset()
    #     r = (len(e.next_users) + np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2,3,4,5], size=1)*10) * (1/7)
    #     rewards.append(r.item())
    #
    #
    # # Apply the default theme
    # sns.set_theme()
    #
    # # Load an example dataset
    # tips = pd.DataFrame({"iter": list(range(53)), "reward": rewards})
    #
    # # Create a visualization
    # sns.relplot(
    #     data=tips,kind="line",
    #     x="iter", y="reward")
    # # plt.plot(list(range(53)), rewards)
    # plt.show()
    e = OrderEnv2()

    for i, month_data in enumerate(e.data):
        count = 0
        print(f"month {i}")
        for day_data in month_data:
            count += len(day_data)
            print(len(day_data))
        print(f"totally {count}")
        print()

        
    # data = next(e)
    # print(len(data))
    # assert False
    # next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels = data

    # for i in range(10):
    #     day_orders = e.data[0][i]
    #     if len(day_orders) == 0:
    #         continue
    #     next_users, next_orders, next_labels, next_dates, next_succdelv_td, next_is_uav, next_is_redelv, next_levels = zip(
    #         *day_orders)
    #     # print(e.next_users[i])
    #     # print(e.next_orders[i])
    #     # print(e.next_labels[i])
    #     # print(e.next_dates[i])
    #     # print(e.next_succdelv_td[i])
    #     # print(e.next_is_uav[i])
    #     # print(e.next_is_redelv[i])
    #     # print(e.next_levels[i])
    #     # print()

    #     print(len(next_users))
    #     print(len(next_orders))
    #     print(len(next_labels))
    #     print(len(next_dates))
    #     print(len(next_succdelv_td))
    #     print(len(next_is_uav))
    #     print(len(next_is_redelv))
    #     print(len(next_levels))
    #     print()
    # e.reset()
    # for i in range(e.spec.max_episode_steps):
    #     _, r, _, _  = e.step(e.action_space.sample())
    #     print(r)

    # next_users, next_orders, next_labels = e.sample_day()

    # print("PART1:")
    # print("[({}, {}), ...] LEN: {}".format(np.shape(ob[0][0][0]), np.shape(ob[0][0][1]), len(ob[0])))
    #
    # print(ob[0][0][0])
    # print(ob[0][0][1])
    # print()
    #
    # print("PART2:")
    # print("({}, {}, {})".format(np.shape(ob[1][0]), np.shape(ob[1][1]), np.shape(ob[1][2])))
    # print(ob[1][0])
    # print(ob[1][1])
    # print(ob[1][2])
    # print()
    #
    # print("PART3:")
    # print("[({}, {}), ...] LEN: {}".format(np.shape(ob[2][0][0]), np.shape(ob[2][0][1]), len(ob[2])))
    # print(ob[2][0][0])
    # print(ob[2][0][1])
    # print()
    #
    # for i in range(10):
    #     ob, r, d, _ = e.step(e.action_space.sample())
    #     print("iter: {}".format(i))
    #     print("[({}, {}), ...] LEN: {}".format(np.shape(ob[0][0][0]), np.shape(ob[0][0][1]), len(ob[0])))
    #     print("({}, {}, {})".format(np.shape(ob[1][0]), np.shape(ob[1][1]), np.shape(ob[1][2])))
    #     print("[({}, {}), ...] LEN: {}".format(np.shape(ob[2][0][0]), np.shape(ob[2][0][1]), len(ob[2])))
    #     print()
