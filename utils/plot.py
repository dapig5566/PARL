import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import ast
import numpy as np
import os

# statistics = []
# with open("logs\\log_MVDQN_with_pretrain_v1_1675229032.txt", "r") as f:
#     for i in range(20):
#         a = f.readline()
#         print(a)
#     while a != "":
#         r_max = np.nan
#         r_min = np.nan
#         r_mean = np.nan
#         p_ent = np.nan
#         p_loss = np.nan
#         v_loss = np.nan
#
#         for count, i in enumerate(a.split(", ")):
#             p = i.find("\'episode_reward_max\'")
#             if p != -1:
#                 r_max = float(i[p:].split(": ")[-1])
#             p = i.find("\'episode_reward_min\'")
#             if p != -1:
#                 r_min = float(i[p:].split(": ")[-1])
#             p = i.find("\'episode_reward_mean\'")
#             if p != -1:
#                 r_mean = float(i[p:].split(": ")[-1])
#             p = i.find("\'policy_entropy\'")
#             if p != -1:
#                 p_ent = float(i[p:].split(": ")[-1])
#             p = i.find("\'policy_loss\'")
#             if p != -1:
#                 p_loss = float(i[p:].split(": ")[-1])
#             p = i.find("\'vf_loss\'")
#             if p != -1:
#                 v_loss = float(i[p:].split(": ")[-1][:-1])
#             if count > 50:
#                 break
#
#         statistics.append((r_max, r_min, r_mean, p_ent, p_loss, v_loss))
#         a = f.readline()
#
# max_reward, min_reward, mean_reward, entropy, policy_loss, value_loss = zip(*statistics)
#
#
# # df = pd.DataFrame({"iteration": list(range(len(max_reward))),
# #                    "max_return": max_reward,
# #                    "min_return": min_reward,
# #                    "mean_return": mean_reward,
# #                    "policy_entropy": entropy,
# #                    "policy_loss": policy_loss,
# #                    "vf_loss": value_loss})
#
# # df.to_csv("statistics.csv")
# mean_reward = [r for r in mean_reward if not np.isnan(r)]
# # print(np.mean(mean_reward[3000:4000]))
# df = pd.DataFrame({"iteration": list(range(len(mean_reward))),
#                    "mean_return": mean_reward})
#
# df.to_csv("logs\\log_MVDQN_with_pretrain_v1_1675229032.csv")




# if not os.path.exists("models\\experiment_1663688688"):
#     os.mkdir("models\\experiment_1663688688")
#
# sns.set_theme()
# fig = plt.figure()
# sns.lineplot(data=df, x="iteration", y="mean_return")
# plt.savefig("models\\experiment_1663688688\\return_plot.png")

# fig = plt.figure()
# sns.lineplot(data=df, x="iteration", y="policy_entropy")
# plt.savefig("models\\experiment_1663512456\\entropy_plot.png")
#
# fig = plt.figure()
# sns.lineplot(data=df, x="iteration", y="policy_loss")
# plt.savefig("models\\experiment_1663512456\\pilocy_loss.png")
#
# fig = plt.figure()
# sns.lineplot(data=df, x="iteration", y="vf_loss")
# plt.savefig("models\\experiment_1663512456\\value_function_loss.png")


# import pickle as pkl
# root = "D:\\liangxingyuan3\\region_data"
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
#
# x, y, z = pkl.load( open(os.path.join(root, "3d_data.dat"), "rb"))
# ax.scatter3D(x, y, z, 'gray')
# ax.set_xlabel('orders')
# ax.set_ylabel('rounds')
# ax.set_zlabel('couriers')
# ax.view_init(10, 3)
# plt.show()

#######################################
# Draw delivery map
#######################################
# dispatch_map = np.zeros([10, 10])
#
# def construct_map():
#     dispatch_map = np.zeros([10, 10])
#     fig, ax = plt.subplots()
#     # ax = fig.add_axes([0,0,1,1])
#     # ax.set_ylim(0,10000)
#     # ax.set_xlim(-1, 10)
#     for i in range(11):
#         ax.plot([0, 10], [i, i], c='black')
#         ax.plot([i, i], [0, 10], c='black')
#     return dispatch_map, ax
#
# def route_to_map(route, map, t=None):
#     if t is None:
#         for t in route:
#             for pt in t:
#                 map[pt[1], pt[0]] += 1
#     else:
#         for pt in route[t]:
#             map[pt[1], pt[0]] += 1
#
# def fill(ax, x, y, half=None, color='blue'):
#     if half is None:
#         ax.fill_between([x, x + 1], [y, y], [y + 1, y + 1], color=color)
#     else:
#         part, total = half
#         if total == 2:
#             if part == 0:
#                 ax.fill_between([x, x + 0.5], [y, y], [y + 1, y + 1], color=color)
#             elif part == 1:
#                 ax.fill_between([x+0.5, x + 1], [y, y], [y + 1, y + 1], color=color)
#             else:
#                 raise ValueError("not valid part number.")
#         else:
#             raise ValueError("not valid total number.")
#
# def draw_map(route, color_map, name, t=None):
#     map, ax = construct_map()
#     for i in range(2):
#         route_to_map(route["car_{}".format(i)], map, t=t)
#     print(map)
#     for i in range(2):
#         delta = 0
#         if t is None:
#             for t in route["car_{}".format(i)]:
#                 for pt in t:
#
#                     if map[pt[1], pt[0]] == 1:
#                         color = [max(c-delta, 0) for c in color_map[i]]
#                         color = rgb_to_str(*color)
#                         print(color)
#                         fill(ax, pt[0], pt[1], color=color)
#                         delta += 20
#                     else:
#                         color = [max(c - delta, 0) for c in color_map[i]]
#                         color = rgb_to_str(*color)
#                         print(color)
#                         fill(ax, pt[0], pt[1], half=(i, map[pt[1], pt[0]]), color=color)
#                         delta += 20
#         else:
#             for pt in route["car_{}".format(i)][t]:
#                 if map[pt[1], pt[0]] == 1:
#                     color = [max(c - delta, 0) for c in color_map[i]]
#                     color = rgb_to_str(*color)
#                     print(color)
#                     fill(ax, pt[0], pt[1], color=color)
#                     delta += 20
#                 else:
#                     color = [max(c - delta, 0) for c in color_map[i]]
#                     color = rgb_to_str(*color)
#                     print(color)
#                     fill(ax, pt[0], pt[1], half=(i, map[pt[1], pt[0]]), color=color)
#                     delta += 20
#     plt.savefig(name)
#
# def rgb_to_str(r, g, b):
#     r = str(hex(r))[2:]
#     if len(r) == 1:
#         r = "0" + r
#     g = str(hex(g))[2:]
#     if len(g) == 1:
#         g = "0" + g
#     b = str(hex(b))[2:]
#     if len(b) == 1:
#         b = "0" + b
#     print(r, g, b)
#     return "#"+ r+ g + b
#
# route = {
#     "car_0": [[(3, 5)], [(1, 5), (1, 6), (1, 4), (3, 3)], [], [], [], [], []],
#     "car_1": [[], [(1, 6), (0, 5), (4, 6)], [], [], [], [], []]
# }
#
# color_map = [(102,102,255), (255,153,0)]
#
#
# draw_map(route, color_map, name="policy1.png")
# plt.show()
# ''

root = "D:\\liangxingyuan3\\Autonomous Delivery\\multi-view-actor-critic\\logs"
file1 = "20230129_0948.txt"
file2 = "新数据预训练.txt"

def get_data_from_file(file):
    values = []
    with open(file, "r") as f:
        line = f.readline()
        while line != "":
            idx = line.find("acc:")
            if idx != -1:
                values.append(line[idx:].split(", ")[0].split(": ")[1])
            line = f.readline()
    return values


file1_acc = get_data_from_file(os.path.join(root, file1))
file2_acc = get_data_from_file(os.path.join(root, file2))

plot_df = pd.DataFrame({"iter": list(range(len(file2_acc))),
           "acc": file2_acc})

sns.set_theme()
sns.lineplot(plot_df, x="iter", y="acc")
plt.show()