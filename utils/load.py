import os
import git
import numpy as np
import argparse, yaml
from datetime import datetime

import loaders
# Model
import models 



def load_model(model,  data, commit = '', timestamp = '', checkpoint = 'last.ckpt', verbose = False):
    if commit == '':
       repo = git.Repo(search_parent_directories=True)
       git_commit_sha = repo.head.object.hexsha[:7]
    else:
       git_commit_sha = commit
    
    
    check_root =  os.path.join('checkpoints', data, model, git_commit_sha) 
    
    allchckpts = natsorted(
                [
                    fname
                    for fname in os.listdir(check_root)
                ],
            )
    
    if args.timestamp == '':
        chosen = allchckpts[-1]
    else:
        dt = datetime.fromisoformat(timestamp) 
        chosen = min(allchckpts,key=lambda x : abs(datetime.fromisoformat(x) - dt))
    
    checkpoint = os.path.join(check_root, chosen, args.checkpoint)
    
    if verbose:
       print('chosen checkpoint: ' + checkpoint)
    
    with open(f'configs/models/{args.model}.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python dictionary format
        model_config = yaml.load(file, Loader=yaml.FullLoader)
    
    ## define the model and load the checkpoint
    model = getattr(models, model_config['class']).load_from_checkpoint(checkpoint)
    return model 


def loda_data(loader, data, mode = 'train'):
    

