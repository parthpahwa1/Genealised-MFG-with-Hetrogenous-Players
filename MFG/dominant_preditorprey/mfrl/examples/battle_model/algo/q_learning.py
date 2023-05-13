"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from . import base
from . import tools
from . import base2
from . import base3
from . import tools2
from . import tools3


class DQN(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**10, batch_size=64, update_every=5):

        super().__init__(sess, env, handle, name, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[obs, feats], target_q=target_q, acts=actions, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


class MFQ(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(sess, env, handle, name, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))

class MTMFQ(base2.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(sess, env, handle, name, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools2.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            obs, feat, acts, act_prob0, act_prob1, act_prob2, act_prob3, obs_next, feat_next, act_prob_next0, act_prob_next1, act_prob_next2, act_prob_next3, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob0=act_prob_next0, prob1=act_prob_next1, prob2=act_prob_next2, prob3=act_prob_next3)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob0=act_prob0, prob1 = act_prob1, prob2=act_prob2, prob3=act_prob3, acts=acts, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mtmfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mtmfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))


class GenQ_MFG_Dominant(base3.Dominant_ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(sess, env, handle, name, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'use_dominant': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools3.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            data_sample = self.replay_buffer.sample()
            obs = data_sample[0]
            feat = data_sample[1]
            acts  = data_sample[2]

            act_prob0 = data_sample[3]
            act_prob1 = data_sample[4]
            act_prob2= data_sample[5]
            act_prob3 = data_sample[6]

            obs_next  = data_sample[13]
            feat_next = data_sample[14]
            
            act_prob_next0 = data_sample[15]
            act_prob_next1 = data_sample[16]
            act_prob_next2 = data_sample[17]
            act_prob_next3 = data_sample[18]

            rewards = data_sample[25]
            dones = data_sample[26]
            masks = data_sample[27]

            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob0=act_prob_next0, prob1=act_prob_next1, prob2=act_prob_next2, prob3=act_prob_next3)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob0=act_prob0, prob1 = act_prob1, prob2=act_prob2, prob3=act_prob3, acts=acts, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] Dominant Agent LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "DOM_AGENT_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "DOM_AGENT_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))


class GenQ_MFG_Passive(base3.Passive_ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(sess, env, handle, name, use_mf=True, use_dominant=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'use_dominant': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools3.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            data_sample = self.replay_buffer.sample()
            obs = data_sample[0]
            feat = data_sample[1]
            acts  = data_sample[2]

            act_prob0 = data_sample[3]
            act_prob1 = data_sample[4]
            act_prob2= data_sample[5]
            act_prob3 = data_sample[6]

            dominant1_state0 = data_sample[7]
            dominant1_state1 = data_sample[8]
            dominant1_action = data_sample[9]
            dominant2_state0 = data_sample[10]
            dominant2_state1 = data_sample[11]
            dominant2_action = data_sample[12]

            obs_next  = data_sample[13]
            feat_next = data_sample[14]
            
            act_prob_next0 = data_sample[15]
            act_prob_next1 = data_sample[16]
            act_prob_next2 = data_sample[17]
            act_prob_next3 = data_sample[18]

            dominant1_next_state0 = data_sample[19]
            dominant1_next_state1 = data_sample[20]
            dominant1_next_action = data_sample[21]
            dominant2_next_state0 = data_sample[22]
            dominant2_next_state1 = data_sample[23]
            dominant2_next_action = data_sample[24]

            rewards = data_sample[25]
            dones = data_sample[26]
            masks = data_sample[27]

            target_q = self.calc_target_q(
                obs=obs_next,
                feature=feat_next,
                rewards=rewards,
                dones=dones,
                prob0=act_prob_next0,
                prob1=act_prob_next1,
                prob2=act_prob_next2,
                prob3=act_prob_next3,
                dominant1_state0=dominant1_next_state0,
                dominant1_state1=dominant1_next_state1,
                dominant1_action=dominant1_next_action,
                dominant2_state0=dominant2_next_state0,
                dominant2_state1=dominant2_next_state1,
                dominant2_action=dominant2_next_action
            )

            loss, q = super().train(
                state=[obs, feat], 
                target_q=target_q,
                prob0=act_prob0,
                prob1 = act_prob1,
                prob2=act_prob2,
                prob3=act_prob3,
                acts=acts,
                dominant1_state0=dominant1_state0,
                dominant1_state1=dominant1_state1,
                dominant1_action=dominant1_action,
                dominant2_state0=dominant2_state0,
                dominant2_state1=dominant2_state1,
                dominant2_action=dominant2_action,
                masks=masks
            )

            self.update()

            if i % 50 == 0:
                print('[*] Passive Agent LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "PASSIVE_AGENT_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "PASSIVE_AGENT_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))

