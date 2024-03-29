### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import math
import matplotlib.pyplot as plt
import argparse
from sklearn import preprocessing
import numpy as np
import os
plt.switch_backend('agg')

dict = str(os.getcwd()) + '\\RL\\'

"""
Parameters in psychological model
"""
hour_to_forget = 20
memory_scale = 0.8

hour_to_intent = 20
intent_scale = 0.8
intent_c = round(1.0 / hour_to_intent, 2) # 0.05


hour_to_urge = 20
prob_run = 0.1

se_scale = 0.8


"""
Parameters in simulator
"""
def switchInit(arg):
    """" switch binary to 0 and 1 """
    notification_per_day = {
        1: 1,
        2: 2,
        3: 3
    }
    return notification_per_day.get(arg, "Invalid boolean value")


dayOfWeek = 7
notification_per_day = 2
max_decsionPerDay = 12
reward = 1.0

max_notification = notification_per_day * dayOfWeek
max_decsionPerWeek = max_decsionPerDay * dayOfWeek #84
loop_num = 7 / dayOfWeek

fixed_hours = [10, 14]

"""
Parameters in algorithm
"""
parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--gamma', type=float, default=0.8, metavar='G', #1
                    help='discount factor for reward (default: 2)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=100, metavar='N', 
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N', #default=1000 #changing it here doesn't matter, the episode length is determined by is_End function in environment.py
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N', #default=20000
                    help='number of episodes (default: 20000)')
parser.add_argument('--test_episodes', type=int, default=100, metavar='N', #default=20000
                    help='number of testing episodes (default: 20000)')
parser.add_argument('--hidden_size', type=int, default=16, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()

"""
Commonly used functions
"""

def switchBinary(arg):
    """" switch binary to 0 and 1 """
    binary = {
        True: 1,
        False: 0
    }
    return binary.get(arg, "Invalid boolean value")

def mm_normalized(x, min, max):
    return (x - min + 0.0)/(max - min + 0.0)

def onehot_normalized_all(data):
    # data = np.array([5,4,3,1])
    return np.concatenate((getOneHotEncoder('weekday').transform(data[0].reshape(1,-1)).toarray(),
                    getOneHotEncoder('state').transform(data[1].reshape(1, -1)).toarray(),
                    getOneHotEncoder('bs').transform(data[2].reshape(1, -1)).toarray(),
                    getOneHotEncoder('se').transform(data[3].reshape(1, -1)).toarray(), 
                     getOneHotEncoder('regen').transform(data[4].reshape(1, -1)).toarray()), axis=None)

def onehot_normalized(data, name):
    return getOneHotEncoder(name).transform(data).reshape(1,-1).toarray()


def getOneHotEncoder(arg):
    """" get the one-hot encoder """
    weekday = [[1], [2], [3], [4], [5], [6], [7]]
    state = [[0], [1]]
    bs = [[1], [2], [3]]
    se = [[0], [1]]
    regen = [[0], [1]]
    


    enc_weekday = preprocessing.OneHotEncoder()
    enc_weekday.fit(weekday)
    enc_state = preprocessing.OneHotEncoder()
    enc_state.fit(state)
    enc_bs = preprocessing.OneHotEncoder()
    enc_bs.fit(bs)
    enc_se = preprocessing.OneHotEncoder()
    enc_se.fit(se)
    enc_regen = preprocessing.OneHotEncoder()
    enc_regen.fit(regen)

    encoder = {
        'weekday': enc_weekday,
        'state': enc_state,
        'bs': enc_bs,
        'se': enc_se,
        'regen': enc_regen
    }
    return encoder.get(arg, "Invalid feature type")


def draw(data, name):
    """" draw the rewards of all episode in a line """
    plt.plot(data)
    #plt.show()
    plt.savefig(dict + '/figure/' + name + '.png')
    plt.close()
    print("draw the figure done!")
    return

def drawMultiple(datalist, names, title):
    """ draw the curve of multiple algorithms in a plot """

    for data in datalist:
        plt.plot(range(len(data)), data)

    plt.legend(names, loc='upper left')
    #plt.show()
    plt.savefig(title + '.png')
    plt.close()
