import os, sys

sys.path.append(os.path.dirname('..'))
from pathlib import Path
import pickle
import numpy as np


res = {}

gynop_dir = 'res/Gynopticon/trained_models/'

if not os.path.exists(gynop_dir):
    print("path not exist")
    exit(1)
    
def get_val_dub(file_exp):
    dub_val_norm = []
    dub_val_cheat = []
    for vote_file in Path(gynop_dir).glob(file_exp):
        _, _, battles = pickle.load(open(vote_file, 'rb'))

        cheater = battles['cheater']
        dub = battles['battle'][-1]['dubious']
        val = battles['battle'][-1]['validity']
        
        for user_N in dub.keys():
            N = user_N.split('_')[-1]
            if N in cheater:
                dub_val_cheat.append((dub[user_N], val[user_N]))
            else:
                dub_val_norm.append((dub[user_N], val[user_N]))
                
    return {'dub_val_norm': dub_val_norm, 'dub_val_cheat': dub_val_cheat}


res['with_liar'] = get_val_dub('vote-with-liar*')
res['without_liar'] = get_val_dub('vote-without-liar*')

