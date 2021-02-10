#This file provides a Q-Learning solution to CartPole problem in OpenAI Gym.
import gym
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class QLearningAgent():

    def __init__(self):
        self.QTable = np.zeros((6,12,2))
        self.discount_factor = 1

    def policy(self,state):
        return np.argmax(self.QTable[state])

    def updateQValue(self, reward, new_state, i, current_state):
        next_value = np.max(self.QTable[new_state])
        learned_value = reward + self.discount_factor * next_value
        lr = self.learning_rate(i)
        old_value = self.QTable[current_state][action]
        self.QTable[current_state][action] = (1 - lr) * old_value + lr * learned_value

    def learning_rate(self, i):
        return max(0.01, min(1.0, 1.0 - math.log10((i + 1) / 25)))

    def exploration_rate(self, i):
        return max(0.1, min(1, 1.0 - math.log10((i + 1) / 25)))

    def getAction(self, state, i, training=True):
        if np.random.random() < self.exploration_rate(i) and training:
            action = env.action_space.sample()
        else:
            action = self.policy(state)

        return action

class Discretizer():
    def discrete(self, state):
        n_bins = (6,12)
        lower_bounds = [env.observation_space.low[2], -math.radians(50)]
        upper_bounds = [env.observation_space.high[2], math.radians(50)]
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        est.fit([lower_bounds, upper_bounds])
        return tuple(map(int, est.transform([[state[2], state[3]]])[0]))


discretizer = Discretizer()
env_name = "CartPole-v1"
env = gym.make(env_name)

number_of_episodes = 1000
agent = QLearningAgent()

for i in range(number_of_episodes):
    current_state, done = discretizer.discrete(env.reset()), False
    counter = 0
    while not done:
        action = agent.getAction(current_state,i)
        state, reward, done, info = env.step(action)
        new_state = discretizer.discrete(state)
        agent.updateQValue(reward, new_state, i, current_state)
        current_state = new_state

print("Training is done!")
sum = 0
for i in range(100):
    current_state, done = discretizer.discrete(env.reset()), False
    counter = 0
    while not done:
        action = agent.policy(current_state)
        state, reward, done, info = env.step(action)
        current_state = discretizer.discrete(state)
        env.render()
        sum += reward

print(sum/100)