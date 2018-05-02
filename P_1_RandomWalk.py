from numpy.random.mtrand import randint
from urllib3.connectionpool import xrange
import random
import matplotlib.pyplot as plt
import numpy as np

#################
##             ##
##   随机漫步   ##
##             ##
#################


# 原生纯Python实现 ###################

# position = 0
# walk = [position]
# steps = 1000
#
# for i in xrange(steps):
#     step = 1 if random.randint(0, 1) else -1
#     position += step
#     walk.append(position)
#
# plt.plot(walk)
# plt.show()

#####################################


# 使用Numpy实现 ######################

# nsteps = 1000
#
# draws = randint(0, 2, size=nsteps)  # 随机生成1000个0或1
#
# steps = np.where(draws > 0, 1, -1)  # 0表示-1,1表示1,标识向左向右迈出的步子
#
# walk = steps.cumsum()  # 计算累积和,即每一步到达的位置
#
# print(walk)
# print(walk.min())
# print(walk.max())
#
# print((np.abs(walk) >= 10).argmax())  # 向左或向右首次穿越10这个位置
#
# # argmax()并不是很高效,因为它无论如何都会对数组进行完全扫描
#
# plt.plot(walk)
# plt.show()

#####################################


# 一次模拟多个随机漫步 ######################

# 5000人的1000步随机漫步统计
nwarkers = 5000
nsteps = 1000

draws = randint(0, 2, size=(nwarkers, nsteps))

steps = np.where(draws > 0, 1, -1)

walks = steps.cumsum(1)

print('最小位移:{}'.format(walks.min()))
print('最大位移:{}'.format(walks.max()))

# 位移30(包含正负)的最小穿越步数
hits30 = (np.abs(walks) >= 30).any(1)  # 5000人的穿越情况

print('位移超过30的人数:{}'.format(hits30.sum()))  # 位移超过30的人数

traversers = walks[hits30]  # 成功穿越30的人的数据

# print(traversers.shape)

crossing_steps = (np.abs(traversers) >= 30).argmax(1)  # 所有穿越30的人首穿越30使用的步数

print('所有穿越30的人首穿越30使用的步数: {}'.format(crossing_steps))
print('首次穿越30使用步数的平均值: {}'.format(crossing_steps.mean()))  # 首次穿越30使用步数的平均值

#####################################
