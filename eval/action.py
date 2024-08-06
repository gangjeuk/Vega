import numpy as np
import pandas as pd
from typing import *

def simul_one_day(user, env, policy: Callable, **kwargs_for_policy):
    state = env.reset()
    
    num_players = env.num_players
    target_total_match = sum(user.target_match_count) // (env.team_size * 2)
    played_match = -1
    total_played = 0
    
    while played_match != 0:
        played_match = 0
    
        # print("Initial State: \n", state)

        actions = np.random.permutation(range(0, num_players))
        # actions = list(itertools.permutations(range(1, num_players + 1), 1))
        # print("Test actions: \n", actions)

        for i in range(num_players):
            env.render()
            # action = np.random.randint(low=1, high=8 + 1)  # this takes random actions
            action = actions[i]
            act_p = user.loc[(user.index == action)]
            
            # pass player, who played enough
            if (act_p.played_match == act_p.target_match_count).item() is True:
                continue
            next_state, reward, done, info = env.step(action)
            # print("Env step %d:" % (i + 1), action, reward, done, info)
            if env.team_draft_finished():
                # ignore unbalanced match
                if float(reward) < -0.15:
                    continue
                team_draft = env.get_team_draft()

                '''
                cheat_prob = (cheater_user.bot_usage_days / user.bot_usage_days.max()).to_list()
                drop_indices = [random.choices([1, 0], weights=[prob, 1 - prob])[0] for prob in cheat_prob]
            
                for i, idx in enumerate(cheater_user.index):
                    if drop_indices[i] == 0:
                        cheater_user = cheater_user.drop(idx)
                        benign_user = pd.concat([benign_user, user[user.index == idx]], axis = 0)
                '''
                for i in range(3):
                    match_draft = team_draft[i*2:i*2+2]
                    user = policy(user, match_draft, **kwargs_for_policy)

                played_match += 1
                
            if done:
                state = env.reset()
                
        total_played += played_match
            
    env.close()
    
    # Too small number of matches
    # assert total_played > (target_total_match * 0.80)
    
    return user

