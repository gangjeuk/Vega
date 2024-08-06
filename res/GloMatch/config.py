import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, 'data')
model_dir = os.path.join(curr_dir, 'data')

MODEL_CONFIG = {
    "lr": {
        "weight_decay": 0.0,
        "dropout": 0.0,
    },
}
