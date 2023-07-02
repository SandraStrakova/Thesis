### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import pandas as pd
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
        
        self.intent =  round(init.intent_scale ** random.randrange(init.hour_to_intent), 3) #a value between 0.8 and 0.12
        
        self.efficacy = 0.001
        
        
        with open(init.dict +'message_descriptors.json', 'r', encoding='utf-8') as file_handle:
            self.messages = json.load(file_handle)


    def getMemory(self):
        return self.memory
    
        

    ##
    # decision whether run or not run
    ##
    def isRun(self, action, message, state, index):
        """
        Decide whether run or not run based on the current state and message.

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
                    #remove for training
                
                relevant = True
                if se == 1:
                    self.efficacy += 0.35 # added value of exercise
                print('feedback', user_state, state[4])
            
            elif time_lastPA > 12 and user_state == mes_descr: #remove time restriction for training 
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
        
        total_prob = round(self.intent * self.efficacy , 3)
        
        return total_prob, weather_prob, relevant


    def getProb(self, state): #context probability (from data)
        """
        Compute the probability of PA based on weather

        """

        return None

