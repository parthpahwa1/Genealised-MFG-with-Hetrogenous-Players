"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


"""Battle
"""

import argparse
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import magent
import csv 
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools, tools3
from examples.battle_model.senario_battle import battle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq', 'GenQ_MFG'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--oppo1', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq', 'GenQ_MFG'}, help='indicate the first opponent model')
    parser.add_argument('--n_round', type=int, default=500, help='set the total number of games')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=60, help='set the size of map')  # then the amount of agents is 72
    parser.add_argument('--max_steps', type=int, default=5000, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)
    parser.add_argument('--mtmfqp', type=int, choices={0,1,2,3}, default=2, help='set the position of mtmfq')
    parser.add_argument('--gmfqp', type=int, choices={0,1,2,3}, default=3, help='set the position of GenQ_MF')
    
    
    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()
    mtmfq_position = args.mtmfqp
    GenQ_MFG_position = args.gmfqp

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-main'.format(args.algo))
    oppo_model_dir1 = os.path.join(BASE_DIR, 'data/models/{}-main'.format(args.oppo1))

    sess = tf.Session(config=tf_config)
    player_handles = handles[1:]
    models = [spawn_ai(args.algo, sess, env, player_handles[0], args.algo + '-main', args.max_steps), spawn_ai(args.oppo1, sess, env, player_handles[1], args.oppo1 + '-main', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir1, step=args.idx[0])
    adv_type_list = [args.algo, args.oppo1]


    config = {
            'max_len': 2**20,
            'batch_size': 1024,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'use_dominant': True,
            'sub_len': sub_len
        }

    global_replay_buffer = tools3.MemoryGroup(**config)

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle, adv_type_list = adv_type_list, render_every=0)

    win_cnt = {'main': 0, 'opponent1': 0, 'draw': 0}
    total_rewards = []
    with open('storepoints_multibattle_{0}_{1}.csv'.format(args.algo, args.oppo1), 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Game", "Reward 1", "Reward 2"))
    for k in range(0, args.n_round):
        total_rewards = runner.run(0.0, k, win_cnt=win_cnt)
        with open('storepoints_multibattle_{0}_{1}.csv'.format(args.algo, args.oppo1), 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(k, total_rewards[0], total_rewards[1]))

    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3} / DRAW {4}'.format(args.algo, win_cnt['main'] / args.n_round, args.oppo1, win_cnt['opponent1'] / args.n_round, win_cnt['draw']/ args.n_round))





