### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
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

"""" The REINFORCE algorithm with baseline and restriction of notification frequency """

# define one neural network
class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        
        self.action_space = action_space
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_outputs)

    # forward fuction to get the predicted result, in this case is a vector of possiblities for all actions/classes
    def forward(self, inputs):
        x = inputs.to(device=torch.device("cuda")) 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        #action_scores = self.linear2(x) # output
        

        return F.softmax(self.output(x), dim = -1) # -1 to take softmax of last dimension

# objective of the RL algorithm
class REINFORCE():
    def __init__(self, hidden_size, num_inputs, action_space, baseline):
        self.action_space = action_space
        # define the network
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.to(device=torch.device("cuda")) #for the case of GPU
        self.model.load_state_dict(torch.load(init.dict + 'policy_w_feedback.csv')) #load a learned policy
        #self.model.train()
        self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.baseline = baseline
        #self.average_return = np.zeros((init.max_decsionPerWeek))
        self.average_return = np.array(self.generateBaseline(self.baseline))
        self.win_return =  [] # a list of np.array, each np.array has returns from the past episodes
        
       
    def accessModel(self):
        for value in self.model.state_dict():
            print(value, '\t', self.model.state_dict()[value].size())
    
    def accessOptimizer(self):
        for element in self.optimizer.state_dict():
            print(element, '\t', self.optimizer.state_dict()[element])


    ##
    # save and load the model from previous learned policy
    ##
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    ##
    # choose the action based on current state
    ##
    def select_action(self, raw_state):

        # once there is no notification left
        
        time_last_PA = raw_state[1] 
       
            # normalize the state values
        normalized_state = self.normalizeState(raw_state)
        state = torch.Tensor([normalized_state]) 

            # neural network calculates the probability of all actions based on this state
        probs = self.model(Variable(state)).cuda()

        
        if time_last_PA < 12:
            message_index = torch.tensor([[6]], dtype=torch.int32)

        else:
        
            # randomly select from 0 & 1 based on probability given in probs
            message_index = probs.multinomial(4).data # returns an index of the actions, example: tensor([[2, 3, 1, 6]], device='cuda:0') 
            
            # get the probability of selected action
        prob = probs[:, message_index[0, 0].type(torch.int64)].view(1, -1) # Example: tensor([[0.0034, 0.0015, 0.7659, 0.1469, 0.0097, 0.0273, 0.0452]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
                                                                                #action 2 is selected
        
            
            # calculate the log(prob) for the selected action at each state
        log_prob = prob.log()

        return message_index[0], log_prob, probs[0][1]

    def normalizeState(self, state):
        """
        normalize the given state
        :param state: np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType']
              
                  # np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                'weekday', 'hour', 'state', 'bs', 'se', 'regen'] 

        :return: new_state: np.array(['Notification_left' (maximum-normalization), 'Time_from_lastRun'(maximum-normalization),
                                     'Time_from_lastNotifi'(maximum-normalization),'hour'(maximum-normalization),'weekday'(one-hot-encoding),
                                       'state'(one-hot-encoding),'bs'(one-hot-encoding), 
                                       'se'(one-hot-encoding), 'regen'(one-hot-encoding)])

                            changed into weekday, bewolking, weercode, regen for the one-hot-encoding
        """
        # first, perform maximum normalization for continuous variables

        '''
        new_state = np.array([init.mm_normalized(state[0], 0, init.max_notification), 
                              init.mm_normalized(state[1], 0, init.max_decsionPerWeek - 1),
                          init.mm_normalized(state[2], 0, init.max_decsionPerWeek - 1),
                          init.mm_normalized(state[4], 0, 24)])
        '''
        # next, perform one-hot encoding for categorical variables (weekday, state, BS, SE, regen)
        
        #new_state = np.append(new_state, init.onehot_normalized_all(state[[3, 5, 6, 7, 8]]))  
        #new_state =  np.array([init.mm_normalized(state[1], 0, init.max_decsionPerWeek - 1)]) #time from last run
        new_state = np.array([init.mm_normalized(state[1], 0, init.max_decsionPerWeek - 1),
                              int(state[6]),int(state[7])]) #
        

        return new_state

    # update the parameter at the end of each episode
    ##
    def finish_episode(self, rewards, gamma, log_probs, baseline, past_rewards, i_episode):
        R = 0
        policy_loss = []
        returns = []

        # calculate the return
        for r in rewards[::-1]: # loop the rewards from the end to beginning
            R = r + gamma * R # gamma = 0.8
            # input R into returns as the first element
            # at each loop, the R is the return for one action/state pair-> return(s4/a4) = R4 + gamma * R5
            returns.insert(0, R)

        self.updatePastWindow(returns)
        #print returns
        
        returns = torch.tensor(np.array(returns) - np.array(self.getWinAveReturns()))
            #We inserted a baseline function G¯t inside the expectation to reduce the high variance, 
            # using the average of all returns Gt in the past n (i.e., all previous) episodes.
        #print('RETURNS', returns)
        # Loss = - sum(log(policy) * return)
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def generateBaseline(self, val):
        scale = val/init.max_decsionPerWeek
        baseline = []
        val = val

        for i in range(init.max_decsionPerWeek):
            baseline.append(val)
            val = val - scale
        return baseline

    def calculatePastWindow(self, past_rewards):
        window = 100
        if len(past_rewards) >= window:
            return self.generateBaseline(np.mean(past_rewards[-window:]))

        else:
            return self.generateBaseline(np.mean(past_rewards))

    def updatePastWindow(self, returns):
        window = 100
        if len(self.win_return) >= window:
            # delete the first elements in self.win_return
            del self.win_return[0]
            self.win_return.append(np.array(returns))
            return
        else:
            self.win_return.append(np.array(returns))
            return

    def getWinAveReturns(self):
        ''' get average of returns of all previous epidsodes'''
        return sum(self.win_return) / (len(self.win_return) + 0.0)

