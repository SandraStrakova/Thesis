### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import __init__ as init
import numpy as np
import saveInfo
import os
import torch
import decisionPoint
import messageSelector

"""" Some functions to control different running environments of algorithms (differ training and testing; with and without restriction) """


# run with environment restriction, the algorithm is not learning any more
def run_test(agent, name, env, i_run, num_episode):
    """
        the run function for learning algorithms (reinforce_agent and reinforce_restrict_agent)
    """
    episode_rewards = []  # the total reward at each timestamp in one episode
    run_match_notification_reward = []  # the number of run after a notification in each episode
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode


    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(init.args.test_episodes): #- init.args.left_episodes #the left_episodes should come from the return of run_learn (rest of the calendar)

        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        log_probs = []
        rewards = []  # the reward at each step
        rewards_notifi = []  # the reward at each step before the notification was sent out
        current_probs = []  # the probability of sending notification at each step

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            

            if state[4] in init.fixed_hours:
                print('fixed hour')

                #action = torch.tensor([1], dtype=torch.int32)

                message, log_prob, current_prob = agent.select_action(state)
                message = message.cpu()
                get_message = messageSelector.SelectMessage(message.numpy()[0])
                print(get_message)
                # env.step(self, action): Step the environment using the chosen action by one timestep.
                # Return observation (np.array), reward (float), done (boolean), info (dict) """
                state, reward, done, info = env.step(action=1, message=message.numpy()[0])


            else:
                print('other hour', state[4])
                state, reward, done, info = env.step(action=0, message=None)

            # save all the rewards in this episode
            rewards.append(reward)

            # append rewards only before notification was sent out
            #if info['notification'] > 0 or action.numpy()[0] == 1:
            #rewards_notifi.append(reward)
            

            # once this episode finished
            if done:
                print("         Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                notification_left.append(getNotificationLeft(info['calendars'], i_episode))

                #wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                #wrong_notification.append(wrong)
                #extra_wrong_notification.append(extra_wrong)

                """save detailed information of the current week into files"""
                #saveInfo.saveInternal(info['calendars'][i_episode], i_run, i_episode, name)
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name, num_episode)

                break

    print ("One testing run done!")

    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification, agent


# run without environment restriction for rule-based algorithms
def run_random(agent, name, env, i_run, num_episode):
    """
            the run function for random-setup algorithms (random_week and random_day)
    """
    episode_rewards = []  # the total reward at each timestamp in one episode
    # average_rewards = []    # the average reward for episodes by now

    run_match_notification_reward = []  # the number of run after a notification in each episode
    # ave_run_match_notification = []  # the average number of run after a notification for episodes by now
    notification_left = [] # the number of notification left at the end of each episode
    
    #wrong_notification = [] # the number of wrong notification in each episode
    #extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode

    """" training: loop the episodes """
   
    for i_episode in range(num_episode):


        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        rewards = []  # the reward at each step
        # rewards_notifi = []  # the reward at each step before the notification was sent out

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
    
            
            action, message = agent.select_action(state)
            
            action = action.cpu()
            
            if message == None:
                state, reward, done, info = env.step(action=action.numpy()[0], message=None)
           

            # env.step(self, action): Step the environment using the chosen action by one timestep.
            # Return observation (np.array), reward (float), done (boolean), info (dict) """
            #state, reward, done, info = env.step(action.numpy()[0])
            else:
                message = message.cpu()
                state, reward, done, info = env.step(action=action.numpy()[0], message=message.numpy()[0])
            #print("info: ", info)

            # save all the rewards in this episode
            rewards.append(reward)

            # once this episode finished
            if done:
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                # average_rewards.append(sum(episode_rewards) / (len(episode_rewards) + 0.0))
                # average_rewards_window = meanInWindow(episode_rewards, 20)

                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                # ave_run_match_notification.append(sum(run_match_notification_reward) / (len(run_match_notification_reward) + 0.0))

                notification_left.append(getNotificationLeft(info['calendars'], i_episode))
                #wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                #wrong_notification.append(wrong)
                #extra_wrong_notification.append(extra_wrong)
                
                """save learning detailed information of the current week into files"""
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name, init.args.test_episodes)

                break

        """save random policy and initial parameters """
        saveInfo.saveParameter(env, agent, name, i_run)

    print ("One random run done!")
    return episode_rewards, notification_left


def run_learn_double(agent, name, env, i_run, num_episode):
    """
        learn on top of a saved policy
    """
    episode_rewards = []  # the total reward at each timestamp in one episode
    # average_rewards = []    # the average reward for episodes by now

    run_match_notification_reward = []  # the number of run after a notification in each episode
    # ave_run_match_notification = []  # the average number of run after a notification for episodes by now
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode

    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(num_episode):

        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        log_probs = []
        rewards = []  # the reward at each step
        rewards_notifi = []  # the reward at each step before the notification was sent out
        current_probs = []  # the probability of sending notification at each step

        #sent_feedback = False

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            
            #if state[4] in init.fixed_hours: #and sent_feedback == False):
                #print('fixed hour')

                #action = torch.tensor([1], dtype=torch.int32)
                #print('STATE', state)
                #print(env.calendars[i_episode].index_in_data)
            message, log_prob, current_prob = agent.select_action(state)
                
            message = message.cpu()
            message = message.numpy()[0]

                #if message == 6:
                  #  sent_feedback = True
                #else:
                  #  sent_feedback = False

                # env.step(self, action): Step the environment using the chosen action by one timestep.
                # Return observation (np.array), reward (float), done (boolean), info (dict) """
            state, reward, done, info = env.step(action=1, message=message)

                    # save all the rewards in this episode
            current_probs.append((t, current_prob))
            log_probs.append(log_prob)
            rewards.append(reward)  
            
            '''
            else:
                print('other hour/no more messages', state[4])
                state, reward, done, info = env.step(action=0, message=None)
                rewards.append(reward)
            '''

            # once this episode finished
            if done:
                print("Episode: {}, message: {}, reward: {}".format(i_episode,message, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                # average_rewards.append(sum(episode_rewards) / (len(episode_rewards) + 0.0))
                # average_rewards_window = meanInWindow(episode_rewards, 20)

                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                # ave_run_match_notification.append(sum(run_match_notification_reward) / (len(run_match_notification_reward) + 0.0))
                
                notification_left.append(getNotificationLeft(info['calendars'], i_episode))
                
                #wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                #wrong_notification.append(wrong)
                #extra_wrong_notification.append(extra_wrong)

                """save detailed information of the current week into files"""
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name, num_episode)

                break

        # update the policy at the end of episode
        agent.finish_episode(rewards, init.args.gamma, log_probs, [], i_episode) # instead of baseline, there was 3.5 
                                                                                        #(but baseline is not used here, using avg returns instead)
        saveInfo.savePolicy(agent, name, i_run, i_episode)

    print ("One learning run done!")

    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification, agent

def getReward(calenders, i_episode):
    """
    Calculate the reward if people run after they receive the notification
    :param calenders: the whole calenders of an environment
    :param i_episode: the index of current week
    :return: how many notification was left & how many reward can get
    """

    reward = 0.0

    # add reward in if decision_point.isRun == decision_point.isNotification == True
    for index in range(init.max_decsionPerWeek):
        decision_point = calenders[i_episode].getGrid(index)

        if decision_point.getRun() and decision_point.getNotification():
            reward = reward + 1.0

    return reward

def getWrongNotification(calenders, i_episode):
    """
        Calculate how many of notifications were sent after a user has been run during the day.
        :param calenders: the whole calenders of an environment
        :param i_episode: the index of current week
        :return: how many notification was left & how many reward can get
        """
    
    wrong_notification = 0.0
    extra_wrong_notification = 0.0
    
    # add reward in if decision_point.isRun == decision_point.isNotification == True
    for index in range(init.max_decsionPerWeek):
        decision_point = calenders[i_episode].getGrid(index)
        
        if (index+1) % init.max_decsionPerDay != 0 and decision_point.getRun():
            # calculate which date it is now
            date = index / init.max_decsionPerDay
            # for all decision points after this run before next day, if there is a notification sent
            for next_index in range (index+1, int((date+1) * init.max_decsionPerDay)):
                if calenders[i_episode].getGrid(next_index).getNotification():
                    wrong_notification = wrong_notification + 1.0
        
            if calenders[i_episode].getGrid(index+1).getNotification():
                    extra_wrong_notification = extra_wrong_notification + 1.0

    return wrong_notification, extra_wrong_notification


# return the notificatoin left in this episode
def getNotificationLeft(calenders, i_episode):
    return calenders[i_episode].getNotificationLeft()

# compute the mean of a window
def meanInWindow(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
