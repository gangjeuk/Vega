import random
import os
import numpy as np
import pandas as pd
import pickle

from ContationDataset.data import botlist, friend, bot_usage_days, user_used_bot
from ContationDataset.data import START_DAY, END_DAY, ONE_DAY
import utils
from eval.match import Env
from res.GloMatch.config import MODEL_CONFIG
from res.Gynopticon.simul import simulate_with_liar, simulate_without_liar
import plot

debug_mode = utils.Config['debug']


bot_days = bot_usage_days
bot_days = pd.DataFrame({'id':bot_days.keys(), 'bot_usage_days':bot_days.values()})

# Friend network list
friend_net = set(friend.actor) | set(friend.target)
friend_net = pd.DataFrame(data=friend_net, columns=['id'])

# Generate Virtual user
user = pd.merge(friend_net, bot_days, how='left', on='id')

# Match participation
target_match_count = pd.DataFrame(np.random.randint(low=1, high=10, size=len(user)))

# User data
user = user.fillna(0)
user['target_match_count'] = target_match_count
user['played_match'] = 0
if debug_mode:
    user = user[:1000]

# Configuration
conf = dict()
conf["dataset"] = "ball"
conf["label"] = "win"
conf["algo"] = "lr"
conf["device"] = "cpu"
conf["num_players"] = len(user)
conf["team_size"] = 3
conf["num_features"] = 19
conf["num_classes"] = 1

conf.update(MODEL_CONFIG[conf["algo"]])

env = Env(conf)

def moba_policy(user, team_draft, **kwargs):
    user.loc[((user.index.isin(team_draft)), 'played_match')] += 1
    benign_user = user.loc[(user.index.isin(team_draft) & user.used_bot.eq(False))]
    cheater_user = user.loc[(user.index.isin(team_draft) & user.used_bot.eq(True))]

    if kwargs['with_liar'] is True:
        benign_picked = pd.DataFrame(random.sample(utils.res['without_liar']['dub_val_norm'], len(benign_user)), columns=['dub', 'val'])
        cheater_picked = pd.DataFrame(random.sample(utils.res['without_liar']['dub_val_cheat'], len(cheater_user)), columns=['dub', 'val'])
    else:
        benign_picked = pd.DataFrame(random.sample(utils.res['without_liar']['dub_val_norm'], len(benign_user)), columns=['dub', 'val'])
        cheater_picked = pd.DataFrame(random.sample(utils.res['without_liar']['dub_val_cheat'], len(cheater_user)), columns=['dub', 'val'])
    
    benign_picked.index = benign_user.index 
    cheater_picked.index = cheater_user.index 
    
    benign_user.update(benign_user.val + benign_picked.val)
    benign_user.update(benign_user.dub + benign_picked.dub)
    
    cheater_user.update(cheater_user.val + cheater_picked.val)
    cheater_user.update(cheater_user.dub + cheater_picked.dub)           
    
    user.update(cheater_user)
    user.update(benign_user)
    
    return user

def action_policy(user, team_draft, **kwargs):
    user.loc[((user.index.isin(team_draft)), 'played_match')] += 1
    benign_user = user.loc[(user.index.isin(team_draft) & user.used_bot.eq(False))]
    cheater_user = user.loc[(user.index.isin(team_draft) & user.used_bot.eq(True))]    
    
    if kwargs['with_liar'] is True:          
        benign, cheater, _ = simulate_with_liar(1,3,0.8,len(benign_user), len(cheater_user))
    else:
        benign, cheater, _ = simulate_without_liar(1,3,0.8,len(benign_user), len(cheater_user))
    benign_picked = pd.DataFrame(benign.values(), columns=['dub', 'val'])
    cheater_picked = pd.DataFrame(cheater.values(), columns=['dub', 'val'])

    benign_picked['voted_to_untrue'] = np.where(benign_picked.val.le(0.5), 1, 0)
    cheater_picked['voted_to_untrue'] = np.where(cheater_picked.val.le(0.5), 1, 0)
    benign_picked['voted_to_cheat'] = np.where(benign_picked.dub.ge(0), 1, 0)
    cheater_picked['voted_to_cheat'] = np.where(cheater_picked.dub.ge(0), 1, 0)

    benign_picked.index = benign_user.index
    cheater_picked.index = cheater_user.index
    
    benign_user.update(benign_user + benign_picked)   
    cheater_user.update(cheater_user + cheater_picked)

    user.update(benign_user)
    user.update(cheater_user)
    
    return user


def simulate_moba(begin_date, end_date, user, with_liar):
    from eval.moba import simul_one_day
    sav_dir = f'data/match/moba_res_withliar-{with_liar}_debugmode-{debug_mode}.pickle'
    # Set variables for simulation
    user['dub'] = .0
    user['val'] = .0   
    try:
        user = pickle.load(open(sav_dir, 'rb'))
    except: 
        # Do simulation
        while begin_date < end_date:
            next_date = begin_date + ONE_DAY
            user['used_bot'] = False
            
            used_bot = user_used_bot(begin_date, next_date)
            
            user.loc[user.id.isin(used_bot), 'used_bot'] = True
            simul_one_day(user, env, moba_policy, with_liar=with_liar)
            begin_date = next_date
        pickle.dump(user, open(sav_dir, 'wb'))
    return user


def simulate_action(begin_date, end_date, user, with_liar):
    from eval.action import simul_one_day
    sav_dir = f'data/match/action_res_withliar-{with_liar}_debugmode-{debug_mode}.pickle'
    # Set variables for simulation
    user['dub'] = .0
    user['val'] = .0    
    user['voted_to_untrue'] = 0
    user['voted_to_cheat'] = 0
    
    try:
        user = pickle.load(open(sav_dir, 'rb'))
    except:
        # Do simulation
        while begin_date < end_date:
            next_date = begin_date + ONE_DAY
            user['used_bot'] = False
            
            used_bot = user_used_bot(begin_date, next_date)
            
            user.loc[user.id.isin(used_bot), 'used_bot'] = True
            simul_one_day(user, env, action_policy, with_liar = with_liar)
            begin_date = next_date
        
        pickle.dump(user, open(sav_dir, 'wb'))
    return user

if __name__ == '__main__':
    action_user_with_liar = user.copy(deep=True)
    action_user_without_liar = user.copy(deep=True)
    
    moba_user_with_liar = user.copy(deep=True)
    moba_user_without_liar = user.copy(deep=True)
    
    
    action_user_with_liar = simulate_action(START_DAY, START_DAY + ONE_DAY, action_user_with_liar, with_liar=True)
    moba_user_with_liar = simulate_moba(START_DAY, START_DAY + ONE_DAY, moba_user_with_liar, with_liar=True)
    action_user_without_liar = simulate_action(START_DAY, START_DAY + ONE_DAY, action_user_without_liar, with_liar=False)
    moba_user_without_liar = simulate_moba(START_DAY, START_DAY + ONE_DAY, moba_user_without_liar, with_liar=False)
    
    plot.figure1(moba_user_without_liar, moba_user_with_liar)
    plot.figure2(action_user_without_liar, action_user_with_liar)