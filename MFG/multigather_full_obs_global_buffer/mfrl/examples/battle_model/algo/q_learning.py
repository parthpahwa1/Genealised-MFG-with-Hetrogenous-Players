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
import copy


class DQN(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**12, batch_size=256, update_every=5):

        super().__init__(sess, env, handle, name, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, opponent_buffer=None):

        if opponent_buffer is not None:
            self.replay_buffer = copy.deepcopy(opponent_buffer)
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        total_loss=0
        for i in range(batch_num):

            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[obs, feats], target_q=target_q, acts=actions, masks=masks)

            total_loss += loss
            self.update()

            if i % 50 == 0:
                print('[*] {0} Agent LOSS:'.format(self.name), loss, '/ Q:', q)
        
        if opponent_buffer is not None:
            self.replay_buffer = None
        
        return total_loss/(batch_num + 1e-5)

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
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**12, batch_size=256):
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

    def train(self, opponent_buffer=None):

        if opponent_buffer is not None:
            self.replay_buffer = copy.deepcopy(opponent_buffer)
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()
        total_loss=0
        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks)

            total_loss += loss
            self.update()

            if i % 50 == 0:
                print('[*] {} Agent LOSS:'.format(self.name), loss, '/ Q:', q)
        if opponent_buffer is not None:
            self.replay_buffer = None
        return total_loss/(batch_name + 1e-5)

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
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**12, batch_size=256):
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

    def train(self, opponent_buffer=None):

        if opponent_buffer is not None:
            self.replay_buffer = copy.deepcopy(opponent_buffer)
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()
        total_loss = 0
        print('MTMFQ:', batch_name)
        for i in range(batch_name):
            obs, feat, acts, act_prob0, act_prob1, act_prob2, act_prob3, obs_next, feat_next, act_prob_next0, act_prob_next1, act_prob_next2, act_prob_next3, rewards, dones, masks = self.replay_buffer.sample()

            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob0=act_prob_next0, prob1=act_prob_next1, prob2=act_prob_next2, prob3=act_prob_next3)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob0=act_prob0, prob1 = act_prob1, prob2=act_prob2, prob3=act_prob3, acts=acts, masks=masks)
            
            total_loss += loss
            self.update()

            if i % 50 == 0:
                print('[*] {0} Agent LOSS:'.format(self.name), loss, '/ Q:', q)
        if opponent_buffer is not None:
            self.replay_buffer = None
        return total_loss/(batch_name + 1e-5)

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



class GenQ_MFG(base3.FullObserve_ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**12, batch_size=256):
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

    def train(self, opponent_buffer=None):

        if opponent_buffer is not None:
            self.replay_buffer = copy.deepcopy(opponent_buffer)
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()
        total_loss = 0
        print('GenQ_MFG:', batch_name)
        for i in range(batch_name):
            data_sample = self.replay_buffer.sample()
            obs = data_sample[0]
            feat = data_sample[1]
            acts  = data_sample[2]

            act_prob0 = data_sample[3]
            act_prob1 = data_sample[4]

            type0_obs = data_sample[5]
            type1_obs = data_sample[6]
            type0_feat = data_sample[7]
            type1_feat = data_sample[8]

            obs_next  = data_sample[9]
            feat_next = data_sample[10]
            
            act_prob_next0 = data_sample[11]
            act_prob_next1 = data_sample[12]

            type0_next_obs = data_sample[13]
            type1_next_obs = data_sample[14]
            type0_next_feat = data_sample[15]
            type1_next_feat = data_sample[16]

            rewards = data_sample[17]
            dones = data_sample[18]
            masks = data_sample[19]

            target_q = self.calc_target_q(
                obs=obs_next,
                feature=feat_next,
                rewards=rewards,
                dones=dones,
                prob0=act_prob_next0,
                prob1=act_prob_next1,
                type0_state0=type0_next_obs,
                type0_state1=type0_next_feat,
                type1_state0=type1_next_obs,
                type1_state1=type1_next_feat
            )

            loss, q = super().train(
                state=[obs, feat], 
                target_q=target_q,
                prob0=act_prob0,
                prob1 = act_prob1,
                acts=acts,
                type0_state0=type0_obs,
                type0_state1=type0_feat,
                type1_state0=type1_obs,
                type1_state1=type1_feat,
                masks=masks
            )
            total_loss += loss
            self.update()

            if i % 50 == 0:
                print('[*] {0} Agent LOSS:'.format(self.name), loss, '/ Q:', q)
        if opponent_buffer is not None:
            self.replay_buffer = None
        return total_loss/(batch_name + 1e-5)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "GEN_MFQ_AGENT_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "GEN_MFQ_AGENT_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))

