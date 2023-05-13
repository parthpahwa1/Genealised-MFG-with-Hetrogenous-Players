"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


from . import ac
from . import q_learning

AC = ac.ActorCritic
MFAC = ac.MFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ
MTMFQ = q_learning.MTMFQ
GenQ_MFG_DOM = q_learning.GenQ_MFG_Dominant
GenQ_MFG_Passive = q_learning.GenQ_MFG_Passive

def spawn_ai(algo_name, sess, env, handle, human_name, max_steps, isDom=False):
    if algo_name == 'GenQ_MFG':
        if isDom:
            model = GenQ_MFG_DOM(sess, human_name, handle, env, max_steps, memory_size=80000)
        else:
            model = GenQ_MFG_Passive(sess, human_name, handle, env, max_steps, memory_size=80000)
            
    elif algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mtmfq':
        model = MTMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    return model
