### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import torch
import pandas as pd
import numpy as np
import os
import RESTRICT_mssg_baseline
import fixed_day as fixed_day
import environment 
import __init__ as init
import run_save
import saveInfo
from torch.autograd import Variable
import torch.nn.utils as utils


#from multiprocessing import Process, Manager, Value, Lock
import time

DISPLAY_REWARD_THRESHOLD = 100  
RENDER = False

EVAL = True

"""" set up the dictionary for saving """

# This is an example main function for running one simulation. You may set up your own code for running different experiments (and multiple jobs).
def oneEnvironment(i_run):

    """read Context info from documents and put them in df"""
    input = os.path.expanduser(init.dict + "state_12.csv")   #the file structure was kept the same for ease of reading but the other variables won't be used
    
    df = pd.read_csv(input, encoding='utf-8-sig', sep=';').dropna()
    
    
    """" set up the training environment """

    if EVAL:
        env = environment.HumanEnv(init.args.test_episodes, df)
        
    else:
        env = environment.HumanEnv(init.args.num_episodes, df) 
    
    # random seed
    env.seed(init.args.seed)
    torch.manual_seed(init.args.seed)
    np.random.seed(init.args.seed)

    # delete the wrap
    env = env.unwrapped
       

    if EVAL:
            #Random_fix algorithm: send notification with maximal init.maxnotification/7 per day at fixed times
            agent_fix = fixed_day.Random()
            raw_rewards_f, notification_left_f = run_save.run_random(agent_fix, 'fix_test', env, i_run, init.args.test_episodes)
            saveInfo.saveTofile(raw_rewards_f, "fix_test", i_run)
            #saveInfo.saveTofile(notification_left_f, "fix_train_notification", i_run)
            #saveInfo.saveTofile(wrong_n_f, "fix_train_wrong", i_run)
            #saveInfo.saveTofile(extra_wrong_n_f, "fix_train_extra_wrong", i_run)
   
       
        
    #baseline = 3.5 #a baseline is subtracted from the obtained return while calculating the gradient.
    if EVAL:
            #Evaluation

            agent_restrict_win_test = RESTRICT_mssg_baseline.REINFORCE(init.args.hidden_size, 3, env.message_space) 
            raw_rewards_rw_test, notification_left_rw_test, wrong_n_rw_test, extra_wrong_n_rw_test, agent_restrict_learned = run_save.run_test(agent_restrict_win_test, 'restrict_win_test',env, i_run, init.args.test_episodes)
            saveInfo.saveTofile(raw_rewards_rw_test, "restrict_win_test", i_run)
            #saveInfo.saveTofile(notification_left_rw_test, "restrict_win_notification_test", i_run)
            #saveInfo.saveTofile(wrong_n_rw_test, "restrict_win_wrong_test", i_run)
            #saveInfo.saveTofile(extra_wrong_n_rw_test, "restrict_win_extra_wrong_test", i_run)

    else:
            # REINFORCE_restrict algorithm: send notification with maximal init.max_notification
            agent_restrict_win = RESTRICT_mssg_baseline.REINFORCE(init.args.hidden_size, 3, env.message_space) 
            raw_rewards_rw_train, notification_left_rw_train, wrong_n_rw_train, extra_wrong_n_rw_train, agent_restrict_learned = run_save.run_learn_double(agent_restrict_win, 'restrict_win',env, i_run, init.args.num_episodes)
            saveInfo.saveTofile(raw_rewards_rw_train, "restrict_win_train", i_run)
            #saveInfo.saveTofile(notification_left_rw_train, "restrict_win_notification", i_run)
            #saveInfo.saveTofile(wrong_n_rw_train, "restrict_win_wrong", i_run)
            #saveInfo.saveTofile(extra_wrong_n_rw_train, "restrict_win_extra_wrong", i_run)
    

    



    env.close()
    
    
    

if __name__ == '__main__':
    oneEnvironment(3)
