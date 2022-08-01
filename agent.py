import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import neuroEnv




actionSpace = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
slideroffset=[0,0,0,0,0,0,0,0,0,0,]

# slider 0 -> +, -
# slider 1 


stateSpace = [0,1]

Rewards = [-1,1]


class Conducter():

    def __init__(self, buckets=(2,20), 
                num_episodes=50, min_lr=0.1, 
                min_epsilon=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        # environment is stochastic for now but will change soon
        # self.env = gym.make('CartPole-v0')
        
        # This is the action-value function being initialized to 0's
        self.Q_table = np.zeros(self.buckets)

        # [position, velocity, angle, angular velocity]
        # self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        # self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]
        
        #
        self.steps = np.zeros(self.num_episodes)

    # This is the policy function (Going to be completely random)


    def choose_action(self, state):
        """
        Implementation of e-greedy algorithm. Returns an action (0 or 1).
        
        Input:
        state (tuple): Tuple containing 4 non-negative integers within
                       the range of the buckets.
        
        Output:
        (int) Returns either 0 or 1
        """
        if (np.random.random() < self.epsilon):
            return np.random.choice(actionSpace, 1)[0] 
        else:
            return np.argmax(self.Q_table[state])

    def get_epsilon(self, t):
        """Gets value for epsilon. It declines as we advance in episodes."""
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring
        # hardcoded for now but need to see how the gradial decent can help
        return 0.5
    

    def update_q(self, state, action, reward, new_state):
        """
        Updates Q-table using the rule as described by Sutton and Barto in
        Reinforcement Learning.
        """
        self.Q_table[state][action] += (self.learning_rate * 
                                        (reward 
                                         + self.discount * np.max(self.Q_table[new_state]) 
                                         - self.Q_table[state][action]))
        print(self.Q_table[state][action])

    def train(self,driver,qq):
        """
        Trains agent making it go through the environment and choose actions
        through an e-greedy policy and updating values for its Q-table. The 
        agent is trained by default for 500 episodes with a declining 
        learning rate and epsilon values that with the default values,
        reach the minimum after 198 episodes.
        """
        # Looping for each episode

        for e in range(self.num_episodes):
            # Initializes the state
            # current_state = np.random.choice(stateSpace, 1)[0] 
            abState=qq.get()
            print("abState")
            print(abState)
            # abState = neuroEnv.ab_result[0][49] 
            # abStateold = neuroEnv.ab_result[0][49] 
            # # print(abState)
            if abState > 0.7:
                current_state=1
            else:
                current_state=0

            self.learning_rate = 0.5
            self.epsilon = self.get_epsilon(e)
            done = False
            timesteps=0
            
            # Looping for each step
            while not done:
                timesteps+=1
                self.steps[e] += 1
                # Choose A from S
                action = self.choose_action(current_state)
                if(action !=20):
                    print("action")
                    print(action)
                    sliderno=action%10
                    increase=int(action / 10)
                    print("sliderno")
                    print(sliderno)
                    print("increase")
                    print(increase)
                    # Take action
                    # obs, reward, done, _ = self.env.step(action)
                    slider=driver.find_element("id", "s"+str(sliderno))
                    ac = ActionChains(driver)
                    if(increase):
                        slideroffset[sliderno]-=20
                        if(slideroffset[sliderno]<=-50):
                            slideroffset[sliderno]= -50                       
                    else:
                        slideroffset[sliderno]+=20
                        if(slideroffset[sliderno]>=50):
                            slideroffset[sliderno]=50                      


                    print(slideroffset)
                    ac.move_to_element(slider).move_by_offset(0,int(slideroffset[sliderno])).click().perform()
                    print("move")
                    print("episode")
                    print(e)
                    print("timestep")
                    print(timesteps)

                    # print(abData)
                time.sleep(2)
                # ac.move_to_element(slider).move_by_offset(0,-50).click().perform()
                
                # abState = neuroEnv.ab_result[0][49]
                # print("new")
                # print(abState)
                # print(abState==abStateold)
                abState=qq.get()
                print("abState")
                print(abState)
                if abState > 0.7:
                    new_state=1
                    reward = 1
                else:
                    new_state=0
                    reward = -1
                # new_state = np.random.choice(stateSpace,1)[0]
                # reward = 1
                
                
                done = new_state == 0
                    
                # Update Q(S,A)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
                if self.steps[e] > 25:
                    break
                # We break out of the loop when done is False which is
                # a terminal state.
        np.savetxt('qTable.csv', self.Q_table, delimiter=',') 
        print('Finished training!')

    def plot_learning(self):
        """
        Plots the number of steps at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        sns.lineplot(range(len(self.steps)),self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()
        t = 0
        # for i in range(self.num_episodes):
        #     if self.steps[i] == 200:
        #         t+=1
        # print(t, "episodes were successfully completed.")
            
        return t   
    def selectRandomAction(self):

        return np.random.choice(actionSpace, 1)[0]


