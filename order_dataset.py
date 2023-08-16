import torch.utils.data as data
import numpy as np
import torch
from env import OrderEnv2
from config import USER_INDEX, INDEX_USER, num_hist


class OrderDataset(data.Dataset):
    def __init__(self, i, max_length=None, user_list=None, split="train", data=None, user_data=None):
        
        if data is not None:
            self.data = data[i]
        else:
            env = OrderEnv2()
            self.data = env.data[i]
        
        self.next_users = []
        self.next_orders = []
        self.next_labels = []
        self.next_dates = []
        self.next_levels = []
        for data in [self.data]:
            for i in range(len(data)):
                day_orders = data[i]
                if len(day_orders) == 0:
                    continue
                next_users, next_orders, next_labels, next_dates, _, _, _, next_levels = zip(*day_orders)
                self.next_users.extend(next_users)
                self.next_orders.extend(next_orders)
                self.next_labels.extend(next_labels)
                self.next_dates.extend(next_dates)
                self.next_levels.extend(next_levels)
        
        # print(len(self.next_users))
        # print(len(self.next_orders))
        # print(len(self.next_labels))
        # print(len(self.next_dates))
        # print(len(self.next_levels))
        # print()
        if split == "train":
            self.next_users = self.next_users[:int(len(self.next_users)*0.8)]
            self.next_orders = self.next_orders[:int(len(self.next_orders)*0.8)]
            self.next_labels = self.next_labels[:int(len(self.next_labels)*0.8)]
            self.next_dates = self.next_dates[:int(len(self.next_dates)*0.8)]
            self.next_levels = self.next_levels[:int(len(self.next_levels)*0.8)]
        elif split == "test":
            self.next_users = self.next_users[-int(len(self.next_users)*0.2):]
            self.next_orders = self.next_orders[-int(len(self.next_orders)*0.2):]
            self.next_labels = self.next_labels[-int(len(self.next_labels)*0.2):]
            self.next_dates = self.next_dates[-int(len(self.next_dates)*0.2):]
            self.next_levels = self.next_levels[-int(len(self.next_levels)*0.2):]
        else:
            pass
        
        if user_list is not None:
            indices = []
            for i, u in enumerate(self.next_users):
                if u in user_list:
                    indices.append(i)
            # print(len(self.next_users))
            # print(len(self.next_orders))
            # print(len(self.next_labels))
            # print(len(self.next_dates))
            # print(len(self.next_levels))
            # print(indices)
            self.next_users = [self.next_users[i] for i in indices]
            self.next_orders = [self.next_orders[i] for i in indices]
            self.next_labels = [self.next_labels[i] for i in indices]
            self.next_dates = [self.next_dates[i] for i in indices]
            self.next_levels = [self.next_levels[i] for i in indices]

        if max_length is not None:
            self.next_users = self.next_users[:max_length]
            self.next_orders = self.next_orders[:max_length]
            self.next_labels = self.next_labels[:max_length]
            self.next_dates = self.next_dates[:max_length]
            self.next_levels = self.next_levels[:max_length]
        if user_data is not None:
            self.user_data = user_data
        else:
            self.user_data = env.user_data

    
    def get_user_list(self):
        unique_users = set(self.next_users)
        return unique_users
    
    def get_obs(self, idx):
        order_to_dispatch = self.next_orders[idx]
        order_date = self.next_dates[idx]
        uid = USER_INDEX[self.next_users[idx]]
        all_user_hist, all_user_dates, _, _, _ = self.user_data[self.next_users[idx]]
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
        if len(user_hist) == 0:
            user_hist = np.zeros([num_hist, 36])
        elif len(user_hist) < num_hist:
            user_hist = np.concatenate([user_hist, np.zeros([num_hist - len(user_hist), 36])])
        # return self.process_state(self.dispatch_state), \
        #        (order_to_dispatch, user_hist, uid), \
        #        self.get_remained_orders(), \
        #        self.dispatch_map
        order_to_dispatch = torch.from_numpy(order_to_dispatch).float()
        user_hist = torch.from_numpy(user_hist).float()
        uid = torch.tensor(uid, dtype=torch.int)
        lv, ent = self.next_levels[idx]
        label = np.argmax(self.next_labels[idx])

        # if lv <= 2:
        #     label = 7
        
        label = np.eye(7)[label]
        y = torch.from_numpy(label).float()
        # y2 = torch.from_numpy(np.eye(3)[5 - self.next_levels[idx][0]]).float()
        y2 = torch.from_numpy(np.eye(5)[5-lv]).float()
        # y2 = torch.from_numpy(np.eye(5)[0 if self.next_levels[idx][0] > 3 else 1]).float()
        return (order_to_dispatch, user_hist, uid), y, y2

    def __getitem__(self, item):
        return self.get_obs(item)

    def __len__(self):
        return len(self.next_orders)

