import magent
import math
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo.q_learning import GenQ_MFG_Passive
import tensorflow.compat.v1 as tf

map_size = 32
env = magent.GridWorld('battle', map_size=map_size)
handles = env.get_handles()


env.reset()
width = map_size
height = map_size

init_num = map_size * map_size * 0.04

gap = 3
# left
n = init_num
side = int(math.sqrt(n)) * 2
pos = [[], []]
ct = 0
for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
    for y in range((height - side)//2, (height - side)//2 + side, 2):
        pos[ct % 2].append([x, y])
    ct += 1
env.add_agents(handles[0], method="custom", pos=pos[0])
env.add_agents(handles[1], method="custom", pos=pos[1])

# right
n = init_num
side = int(math.sqrt(n)) * 2
pos = [[], []]
ct = 0
for x in range(width//2 + gap, width//2 + gap + side, 2):
    for y in range((height - side)//2, (height - side)//2 + side, 2):
        pos[ct % 2].append([x, y])
    ct += 1
env.add_agents(handles[2], method="custom", pos=pos[0])
env.add_agents(handles[3], method="custom", pos=pos[1])

print(env.get_action_space(handles[1])[0])
print(env.get_reward(handles[1]))
print(env.get_num(handles[1]))
print(env.get_alive(handles[1]))
print(env.get_feature_space(handles[1]))
print(env.get_view_space(handles[1]))

print(env.get_observation(handles[1])[0].shape)
print(env.get_observation(handles[1])[1].shape)
print(env.get_agent_id(handles[0]))
# get_num

tf.reset_default_graph()
tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
tf_config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=tf_config)

# with tf.Session(config=tf_config) as sess: 
agent1 = spawn_ai('GenQ_MFG', sess, env, handles[0], 'GenQ_MFG-dom1', 500, isDom=False)

import sys
import numpy as np
print(type(agent1)==GenQ_MFG_Passive)
temp = tf.repeat([[1,3]], repeats=4, axis=0)
print(sess.run(temp))
tf.print(temp, [temp])
