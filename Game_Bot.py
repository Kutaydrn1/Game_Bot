import gym
import numpy as np

env = gym.make('CartPole-v1')

# Initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Q-learning algorithm
for episode in range(1, 10001):
    state = env.reset()
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, info = env.step(action)
        
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[state, action] = new_value
        
        state = next_state
    
    if episode % 100 == 0:
        print(f"Episode {episode}")

# Visualize the trained bot using the Q-table
for episode in range(3):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
        env.render()
