#This file provides a Deep Q-Learning solution to CartPole problem in OpenAI Gym.
import gym
import random
import numpy as np
from collections import deque
import tensorflow as tf

class DQNAgent():

    def __init__(self,state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size

        self.memory = deque(maxlen=2000)
        self.discount_factor = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.model = self.build_model()
        #self.model = self.load_model("CartPoleDQNModel")
        #If you want to use existing model you can comment the first line and uncomment the second line
        #If you want to use existing model you can comment out training part.

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                target = reward + self.discount_factor*np.amax(self.model.predict(next_state)[0])

            target_values = self.model.predict(state)
            target_values[0][action] = target
            self.model.fit(state, target_values, epochs=1, verbose=0)

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load_model(self, name):
        return tf.keras.models.load_model(name)

    def save_model(self, name):
        self.model.save(name)


env_name = "CartPole-v1"
env = gym.make(env_name)
number_of_episodes = 500
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
agent = DQNAgent(state_size, action_size)

for e in range(number_of_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        #env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        reward = -10 if done else reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("Episode Number: {}, Reward: {}".format(e, time))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)


print("Training is done! Testing will start!")
agent.save_model("CartPoleDQNModel")

agent.epsilon = 0
total_reward = 0
number_of_test_episodes = 100
for t in range(number_of_test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            total_reward += time
            break

print("Average reward of 100 consecutive trials: {}".format(total_reward/number_of_test_episodes))
