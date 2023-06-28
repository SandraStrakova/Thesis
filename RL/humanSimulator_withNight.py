### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
import __init__ as init
import os
import json

""""Consider the influence of night break"""

class Human(object):
    """ The human object, representing a person can decide run or not run.

        Args:
            memory (float): personal memory level of running.
            urge (float): mental motivation to run.
            prob (float): a parameter to balance the real human mental and assumptions/
    """

    def __init__(self):
        self.memory = init.memory_scale ** random.randrange(init.hour_to_forget)  # random initialize #time from last run
        #self.prob = init.prob_run
        
        self.intent =  round(init.intent_scale ** random.randrange(init.hour_to_intent), 3) #a value between 0.8 and 0.12
        
        self.efficacy = 0.001
        
        
        with open(init.dict +'message_descriptors.json', 'r', encoding='utf-8') as file_handle:
            self.messages = json.load(file_handle)


    def getMemory(self):
        return self.memory
    
    def resetIntention(self):
        self.intent =  round(init.intent_scale ** random.randrange(init.hour_to_intent), 3)
        

    ##
    # decision whether run or not run
    ##
    def isRun(self, action, message, state, index):
        """
        :param action: 1 is send notification, 0 is not.
        :param state: np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                          'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'])
        :return: Boolean True = run, False = not run

        """
        prob, weather_prob, relevant = self.computeProb(action,message,state, index)

        print('new self intent: ', self.intent)
        print('prob: ', prob)

        if prob > 1.0:
            print("The probability of running is too high.")
            print('run: ', True, '\n')
            return True, 1.0, relevant
        
        else:
            run = np.random.choice([True, False], 1, p=[prob, 1 - prob])
            print('run: ', run, '\n')
            return run, prob, relevant


    def computeProb(self, action, message, state, index):
       
        
        print('self.intent existing: ', self.intent)


        """Compute the probablity of run. P(PA_t | I_t, SE_t) 
                                          P(PA_t | PA_t-1, I_t, N_t, SE_t)

        Args:
            action (bool): send notification or not, True = sent, False =  not sent.
            state (np.array): ['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                          'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType']
            last_run (bool): run or not in the last time step, True = run, False =  not run.

        Returns:
            prob (float): probability of running, in [0,1].
        """

        for m in self.messages:
            if m['ID'] == message:
                mes_descr = m['descr']  # phase, determinant / feedback 
                break

       
        # state: hour, state, BS, SE (0,1), regen 
        time_lastPA = state[1]
        bs = state[6]  # should increase PA likelihood
        se = state[7]
        user_state = [bs, se]

        if se == 0:
            self.efficacy = random.randrange(6,10) / 100
        if se == 1:
            self.efficacy = random.randrange(1,6) / 100
        


        # when it is the first hour in a day, update intent, so it's not the same as the previous day
        print('INDEX, state1', index, state[1])
        if index % init.max_decsionPerDay == 0:
            self.intent = round(self.intent + init.intent_c * 6, 2) # + 0.3
            print('self.intent after night: ', self.intent)

        else:
            self.intent = self.intent + init.intent_c
        
        
        if bs == 1: 
            self.intent = self.intent * 0.06 #2058/35000 = 0.06
         
        elif bs == 2:
              self.intent = self.intent * 0.16 #5757/35000 = 0.16  
              if self.efficacy == 1: 
                self.efficacy += 0.2
              
        elif bs == 3:
                 self.intent = self.intent * 0.43 # avg week + weekend / 5k*7: 7935+7283/35000 = 0.43
                 if se == 1:
                    self.efficacy += 0.3
        




        if action == 1:        
            
            if time_lastPA < 12 and mes_descr == [4,4]:  # if time_from_lastRun < 12 & got feedback
                
                relevant = True
                if se == 1:
                    self.efficacy += 0.35 # added value of exercise
                print('feedback', user_state, state[4])

            elif time_lastPA > 12 and user_state == mes_descr:
                relevant = True
                self.intent += 0.65
                if se == 1:
                    self.efficacy += 0.3
                print('matching', user_state, state[4])
         
            else:
                relevant = False
                print('not matching ', user_state, mes_descr)
            
        else:
            relevant = None
            print('no message ')
            # no effect on intent
       
       

        if time_lastPA > 0 and time_lastPA <= 12:  # if time_from_lastRun was in the day
            self.intent = 0.001
            print('intent reset to 0.001', time_lastPA)
            #potentially: increase self.efficacy

        if self.efficacy == 0:
            self.efficacy = 0.001
    
        
        
        weather_prob = 0 #self.getProb(state[3:9], action)
        #total_prob = self.memory * self.urge * weather_prob * self.prob * self.preference
        
        total_prob = round(self.intent * self.efficacy , 3)
        
        return total_prob, weather_prob, relevant


    def getProb(self, state, action): #context probability (from data)
        """Compute the probability P(Ct | Rt = 1) / P(Ct) from data.

            data saved in "/Users/Shihan/Desktop/Experiments/PAUL/rl/mylaps/weekday.csv"
                        "/Users/Shihan/Desktop/Experiments/PAUL/rl/mylaps/model/"
                        "/Users/Shihan/Desktop/Experiments/PAUL/rl/knmi/model/"
            Args:
                context (np.array): An array of context info ['Weekday', 'Hour', 'Temperatuur', 'WeerType', 'WindType', 'LuchtvochtigheidType'].

            Returns:
                prob (float): a probability in [0, ?) can be bigger than 1. !!!
            """

        inpath = init.dict

        #weekday = context[:, 1][0]
        #name = str(tuple(context[:, [3,4,5]][0]))
        #data = context[:, [0,1]]

        weekday = state[0]
        name = str(tuple(state[3:])).replace('.0', '')
        index = np.array([False, True, True, False, False, False])
        data = state[index]

        if state[5] == 1:
            prob = 0.4
        else:
            prob = 0.8

        return prob

