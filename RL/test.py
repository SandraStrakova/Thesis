import os
import numpy as np
import pandas as pd
import gym as gym
import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import __init__ as init
import weekCalendar
import humanSimulator_withNight
from random import randrange
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np
import __init__ as init

import random
num = np.arange(0, 17724, 84)
#print(num)

data = 'C:\\Users\\sebas\\Documents\\GitHub\\Thesis\\RL\\state_12.csv'
df = pd.read_csv(data, encoding='utf-8-sig', sep=';').dropna()


message_data = 'C:\\Users\\sebas\\Desktop\\messages.csv'
df_mssg = pd.read_csv(message_data, encoding='unicode_escape',sep=';').dropna()
filter_ = df_mssg[df_mssg['phases'] == 'INITIATION']['message']
#print(df_mssg['message'].head())
print(filter_.iloc[0])
