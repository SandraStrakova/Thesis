### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import torch
import numpy as np
import random
import __init__ as init

class Random:
    def __init__(self, hours=init.fixed_hours):
        """ The object, send a certain number of notifications in a day at fixed times.

                Args:
                    notification_hours (np.array): an array of index representing the hours to send notifications in a day.
            """
        self.notification_hours_list = hours



    def getSave(self):
        """return a list of array"""
        return self.notification_hours_list

    def select_action(self, state):

        if state[4] in self.notification_hours_list: #if it's 10 or 14
                     #state = np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi','weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'])
            #print('state ',state)
            message = np.random.randint(0,6, dtype=int)

            return torch.tensor([1], dtype=torch.int32), torch.tensor([message], dtype=torch.int32)
        else:
            return torch.tensor([0], dtype=torch.int32), None
        
   
    
        

       