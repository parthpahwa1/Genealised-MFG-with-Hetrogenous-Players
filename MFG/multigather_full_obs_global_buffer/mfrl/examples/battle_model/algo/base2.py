"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from magent.gridworld import GridWorld


class ValueNet:
    def __init__(self, sess, env, handle, name, update_every=5, use_mf=False, learning_rate=1e-4, tau=0.005, gamma=0.99):
        # assert isinstance(env, GridWorld)
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 1

        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma

        with tf.variable_scope(name or "ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            self.obs_input = tf.placeholder(tf.float32, (None,) + self.view_space, name="Obs-Input")
            self.feat_input = tf.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')

            if self.use_mf:
                self.act_prob_input0 = tf.placeholder(tf.float32, (None, self.num_actions), name="Act-Prob-Input0")
                self.act_prob_input1 = tf.placeholder(tf.float32, (None, self.num_actions), name="Act-Prob-Input1")
                self.act_prob_input2 = tf.placeholder(tf.float32, (None, self.num_actions), name="Act-Prob-Input2")
                self.act_prob_input3 = tf.placeholder(tf.float32, (None, self.num_actions), name="Act-Prob-Input3")
            
            self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)

            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q, self.obs_embd, self.feat_embd = self._construct_net(active_func=tf.nn.relu)
                self.predict = tf.nn.softmax(self.e_q / self.temperature)
                self.e_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q, _, _ = self._construct_net(active_func=tf.nn.relu)
                self.t_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                    for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - self.e_q_max) * self.mask) / tf.reduce_sum(self.mask)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_net(self, active_func=None, reuse=False):
        conv1 = tf.layers.conv2d(self.obs_input, filters=32, kernel_size=3,
                                 activation=active_func, name="Conv1")
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, activation=active_func,
                                 name="Conv2")
        flatten_obs = tf.reshape(conv2, [-1, np.prod([v.value for v in conv2.shape[1:]])])

        h_obs = tf.layers.dense(flatten_obs, units=256, activation=active_func,
                                name="Dense-Obs")
        h_emb = tf.layers.dense(self.feat_input, units=32, activation=active_func,
                                name="Dense-Emb", reuse=reuse)

        concat_layer = tf.concat([h_obs, h_emb], axis=1)

        if self.use_mf:
            
            prob_emb = tf.layers.dense(self.act_prob_input0, units=64, activation=active_func, name='Prob-Emb0')
            h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob0")
            concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)

            prob_emb = tf.layers.dense(self.act_prob_input1, units=64, activation=active_func, name='Prob-Emb1')
            h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob1")
            concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)

            # prob_emb = tf.layers.dense(self.act_prob_input2, units=64, activation=active_func, name='Prob-Emb2')
            # h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob2")
            # concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)
        
            # prob_emb = tf.layers.dense(self.act_prob_input3, units=64, activation=active_func, name='Prob-Emb3')
            # h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob3")
            # concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)
        
        dense2 = tf.layers.dense(concat_layer, units=512, activation=active_func, name="Dense2")
        # dense2_dropout = tf.layers.dropout(dense2, rate = 0.2, name="Dense2-dropout" )

        dense3 = tf.layers.dense(dense2, units=512, activation=active_func, name="Dense3")
        # dense3_dropout = tf.layers.dropout(dense3, rate = 0.2, name="Dense3-dropout" )
        
        out = tf.layers.dense(dense3, units=64, activation=active_func, name="Dense-Out")

        q = tf.layers.dense(out, units=self.num_actions, name="Q-Value")

        return q, h_obs, h_emb

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob0', 'prob1', 'prob2', 'prob3', 'dones', 'rewards'}
        """
        feed_dict = {
            self.obs_input: kwargs['obs'],
            self.feat_input: kwargs['feature']
        }

        if self.use_mf:
            assert kwargs.get('prob0', None) is not None
            feed_dict[self.act_prob_input0] = kwargs['prob0']

            assert kwargs.get('prob1', None) is not None
            feed_dict[self.act_prob_input1] = kwargs['prob1']
            
            assert kwargs.get('prob2', None) is not None
            feed_dict[self.act_prob_input2] = kwargs['prob2']
            
            assert kwargs.get('prob3', None) is not None
            feed_dict[self.act_prob_input3] = kwargs['prob3']

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        act_idx = np.argmax(e_q, axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]

        target_q_value = kwargs['rewards'] + (1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob0', 'prob1', 'prob2', 'prob3', 'eps'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1]
        }

        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob0', None) is not None
            assert len(kwargs['prob0']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input0] = kwargs['prob0']
            
            assert kwargs.get('prob1', None) is not None
            assert len(kwargs['prob1']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input1] = kwargs['prob1']
            
            assert kwargs.get('prob2', None) is not None
            assert len(kwargs['prob2']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input2] = kwargs['prob2']
            
            assert kwargs.get('prob3', None) is not None
            assert len(kwargs['prob3']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input3] = kwargs['prob3']
        
        actions, obs_embd, feat_embd = self.sess.run(
            [
                self.predict,
                self.obs_embd,
                self.feat_embd,
            ], feed_dict=feed_dict)

        is_greedy = kwargs.get('greedy', False)

        if is_greedy == True:
            actions = np.argmax(actions, axis=1).astype(np.int32)
        else:
            values   = np.repeat(np.arange(actions.shape[1]).reshape((1,actions.shape[1])), actions.shape[0], axis=0)
            actions    = np.array([np.random.choice(v, 1, p=r) for v,r in zip(values, actions)]).astype(np.int32).reshape(-1)
            
        return actions, obs_embd, feat_embd

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob0', 'prob1', 'prob2', 'prob3', 'acts'}

        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.mask: kwargs['masks']
        }

        if self.use_mf:
            assert kwargs.get('prob0', None) is not None
            feed_dict[self.act_prob_input0] = kwargs['prob0']
            
            assert kwargs.get('prob1', None) is not None
            feed_dict[self.act_prob_input1] = kwargs['prob1']
            
            assert kwargs.get('prob2', None) is not None
            feed_dict[self.act_prob_input2] = kwargs['prob2']
            
            assert kwargs.get('prob3', None) is not None
            feed_dict[self.act_prob_input3] = kwargs['prob3']
        
        feed_dict[self.act_input] = kwargs['acts']
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}