"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 10, 'speed': 2,
            'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(2), 'damage': 5, 'step_recover': 0.1, 'kill_reward': 5, 'dead_penalty': -1, 'attack_penalty': -0.2
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 10, 'speed': 2.5,
            'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(0), 'damage': 0, 'step_recover': 0.1, 'kill_reward':10, 'dead_penalty': -1 
        })

    predator_group  = cfg.add_group(predator)
    predator_group2  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)
    prey_group2 = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(predator_group2, index='any')
    b = gw.AgentSymbol(prey_group, index='any')
    d = gw.AgentSymbol(prey_group2, index = 'any')

    cfg.add_reward_rule(gw.Event(a, 'attack', a), receiver=a, value=-0.5)
    cfg.add_reward_rule(gw.Event(c, 'attack', c), receiver=c, value=-0.5)
    
    cfg.add_reward_rule(gw.Event(a, 'kill', a), receiver=a, value=-1)
    cfg.add_reward_rule(gw.Event(c, 'kill', c), receiver=c, value=-1)


    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[0.5, -0.5])
    cfg.add_reward_rule(gw.Event(a, 'attack', d), receiver=[a, d], value=[0.5, -0.5])

    cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=[c, d], value=[0.5, -0.5])
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=[c, b], value=[0.5, -0.5])

    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=[a, c], value=[0.2, -0.2])
    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=[c, a], value=[0.2, -0.2])


    return cfg



# def get_config(map_size):
#     gw = magent.gridworld
#     cfg = gw.Config()

#     cfg.set({"map_width": map_size, "map_height": map_size})
#     cfg.set({"minimap_mode": True})
#     cfg.set({"embedding_size": 10})

#     small = cfg.register_agent_type(
#         "small",
#         {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
#          'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(2),
#          'damage': 2, 'step_recover': 0.1, 'attack_in_group': True,

#          'step_reward': -0.01,  'kill_reward': 0, 'dead_penalty': -1, 'attack_penalty': -0.1,
#          })

#     g0 = cfg.add_group(small)
#     g1 = cfg.add_group(small)
#     g2 = cfg.add_group(small)
#     g3 = cfg.add_group(small)

#     a = gw.AgentSymbol(g0, index='any')
#     b = gw.AgentSymbol(g1, index='any')
#     c = gw.AgentSymbol(g2, index='any')
#     d = gw.AgentSymbol(g3, index='any')

#     # reward shaping to encourage attack
#     cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)
#     cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=a, value=0.3)
#     cfg.add_reward_rule(gw.Event(a, 'attack', d), receiver=a, value=0.4)
#     cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.4)
#     cfg.add_reward_rule(gw.Event(b, 'attack', c), receiver=b, value=0.2)
#     cfg.add_reward_rule(gw.Event(b, 'attack', d), receiver=b, value=0.3)
#     cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=c, value=0.2)
#     cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=0.3)
#     cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=0.4)
#     cfg.add_reward_rule(gw.Event(d, 'attack', a), receiver=d, value=0.2)
#     cfg.add_reward_rule(gw.Event(d, 'attack', b), receiver=d, value=0.3)
#     cfg.add_reward_rule(gw.Event(d, 'attack', c), receiver=d, value=0.4)
#     cfg.add_reward_rule(gw.Event(a, 'kill', b), receiver=a, value=80)
#     cfg.add_reward_rule(gw.Event(a, 'kill', c), receiver=a, value=90)
#     cfg.add_reward_rule(gw.Event(a, 'kill', d), receiver=a, value=100)
#     cfg.add_reward_rule(gw.Event(b, 'kill', a), receiver=b, value=100)
#     cfg.add_reward_rule(gw.Event(b, 'kill', c), receiver=b, value=80)
#     cfg.add_reward_rule(gw.Event(b, 'kill', d), receiver=b, value=90)
#     cfg.add_reward_rule(gw.Event(c, 'kill', d), receiver=c, value=80)
#     cfg.add_reward_rule(gw.Event(c, 'kill', a), receiver=c, value=90)
#     cfg.add_reward_rule(gw.Event(c, 'kill', b), receiver=c, value=100)
#     cfg.add_reward_rule(gw.Event(d, 'kill', a), receiver=d, value=80)
#     cfg.add_reward_rule(gw.Event(d, 'kill', b), receiver=d, value=90)
#     cfg.add_reward_rule(gw.Event(d, 'kill', c), receiver=d, value=100)
#     return cfg
