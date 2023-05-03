### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import __init__ as init
import decisionPoint

class weekMatrix(): #object

    """ The weekMatrix object, representing one week in the calendar

        Args:
            grids (list): a list of grid/decision point
            n_height (int): the number of decision points per day
            n_width (int): the number of days per week
            len = n_width * n_height (int): the total number of decision points in one matrix
            notification_left (int): the number of notifications allowed in one episode
            index_lastnotification (int): the index of last notification


            total_run (int): the number of runs performed in one matrix
            index_lastrun (int): the index of last run

            reward_total (int): the total reward got in one episode
            index_in_data (int): a random index of starting point in real data
    """

    def __init__(self, n_width,  n_height, start_index_in_data, episode, df):
        self.grids = []
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.index_in_data = np.random.randint(0, 3084) #start_index_in_data + episode * init.max_decsionPerWeek 
                                                            #start_index = random_index from randrange(1700)
                                                            #num episodes from init = 200
                                                                #in the generateCalendar function, we loop over the number of episodes (for i in range(num)) -> i becomes the episode
                                                                # max_decsionPerWeek = max_decsionPerDay * dayOfWeek -> 12 * 7 = 84
                                                                # 1700 + 200 * 84 = 1700 + 16800 = 18500

        self.total_run = 0
        self.notification_left = init.max_notification # notification_per_day * dayOfWeek = 2 * 7 = 14
        self.reward_total = 0.0

        for x in range(self.n_width):
            for y in range(self.n_height):
                self.grids.append(decisionPoint.decisionPoint(episode, x, y, self.readContext(self.index_in_data, init.max_decsionPerDay * x + y, df))) 


    def readContext(self, start_index, index, df):
        # ['Weekday', 'Hour', 'Temperatuur', 'WeerType', 'WindType', 'LuchtvochtigheidType']
        state = df.iloc[start_index] # + index]
  
        return np.array(state)

    ##
    # Given the get the information of one decision point
    # return object of decision point
    ##
    def getGrid(self, index):
        #print("index is ", index)	
        #print("len ", len(self.grids))
        return self.grids[index]

    def setTotalRun(self, total_run):
        self.total_run = total_run

    def getTotalRun(self):
        return self.total_run

    def setNotificationLeft(self, notification_left):
        self.notification_left = notification_left

    def getNotificationLeft(self):
        return self.notification_left

    def setTotalReward(self, reward_total):
        self.reward_total = reward_total

    def getTotalReward(self):
        return self.reward_total

    # print the whole calendar
    def printCalendar(self):
        print("--------Here is the calendar info---------")
        for grid in self.grids:
            grid.printInfo()

    # return the data of all decision points in one array
    # example: array([[0, 0, 3, 0, 0, 1, probabilityOfSendNotification], [0, 2, 4, 0, 1, 1], [0, 2, 4, 0, 1, 1]])
    # [index, x, y, condition, notification, run, probabilityOfSendNotification]
    def returnCalendar(self):

        data = np.empty((0, 0))
        for i in range(len(self.grids)):
            if i == 0:
                data = self.grids[i].returnInfo()
            elif i == 1:
                data = np.append([data], [self.grids[i].returnInfo()], axis=0)
            else:
                data = np.append(data, [self.grids[i].returnInfo()], axis=0)
        return data