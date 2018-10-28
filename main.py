import retro
import random
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from keras.models import Sequential
from keras.layers import Dense
from collections import deque
from model import CNN
from retro_wrappers import wrap_deepmind_retro, StochasticFrameSkip

GAMMA = 0.9
MEMORY_SIZE = 500000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
OFFLINE_NETWORK_UPDATE_FREQUENCY = 10000
MODEL_SAVE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EXPLORATION_STEPS = 120000
EPSILON_DECAY = (EPSILON_MAX - EPSILON_MIN) / EXPLORATION_STEPS

PLAY_FROM_WEIGHTS = False

class Agent:
    def __init__(self):
        self.epsilon = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.offline = CNN((4, 84, 84), 6)
        self.online = CNN((4, 84, 84), 6)
        
        if PLAY_FROM_WEIGHTS:
            self.epsilon = 0.0
            self.offline.model.load_weights('online_model.h5')
            self.online.model.load_weights('online_model.h5')

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def act(self, current_state):
        if self.epsilon > np.random.uniform():
            return np.random.randint(6)
        current_state = np.swapaxes(np.expand_dims(current_state, axis=0), 1, 3)
        expected_returns = self.online.model.predict(current_state)[0]
        return np.argmax(expected_returns)

    def replay(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        loss = 0

        for current_state, action, reward, next_state, done in mini_batch:
            current_state = np.swapaxes(np.expand_dims(current_state, axis=0), 1, 3)
            next_state = np.swapaxes(np.expand_dims(next_state, axis=0), 1, 3)

            target = reward
            if not done:
                target = reward + GAMMA * \
                         np.max(self.offline.model.predict(next_state)[0])
            target_f = self.online.model.predict(current_state)
            target_f[0, action] = target
            history = self.online.model.fit(current_state, target_f, epochs=1, verbose=0)
            loss += history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon -= EPSILON_DECAY

        return loss / BATCH_SIZE


if __name__ == '__main__':
    actions = [(), ('LEFT'), ('RIGHT'), ('A'), ('RIGHT', 'A'), ('LEFT','A')]
    action_dict = {():[0,0,0,0,0,0,0,0,0] 
        , ('LEFT'): [0,0,0,0,0,0,1,0,0]
        , ('RIGHT'): [0,0,0,0,0,0,0,1,0]
        , ('A') : [0,0,0,0,0,0,0,0,1]
        , ('LEFT','A'): [0,0,0,0,0,0,1,0,1]
        , ('RIGHT', 'A'): [0,0,0,0,0,0,0,1,1]}
    env = retro.make('SuperMarioBros-Nes', retro.State.DEFAULT)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = wrap_deepmind_retro(env, scale=False)
    agent = Agent()

    total_step = 0
    loss_history = []
    reward_history = []
    current_episode = 0

    while True:
        current_state = env.reset()
        total_reward = 0
        done = False
        current_episode += 1

        while not done:
            if PLAY_FROM_WEIGHTS:
                env.render()
            action_number = agent.act(current_state)
            action = action_dict[actions[action_number]]
            next_state, reward, done, info = env.step(action)

            if reward == 0:
                reward = -.1

            if info['lives'] != 2:
                done = True
                reward = -1

            agent.remember(current_state, action_number, reward, next_state, done)

            current_state = next_state
            total_reward += reward
            total_step += 1

            if total_step > REPLAY_START_SIZE and total_step % TRAINING_FREQUENCY == 0 and not PLAY_FROM_WEIGHTS:
                loss = agent.replay()
                # loss_history.append(loss)
            
            if total_step % OFFLINE_NETWORK_UPDATE_FREQUENCY == 0:
                print('Updating offline network...')
                print('Current epsilon = {0:.2f}'.format(agent.epsilon))
                agent.offline.model.set_weights(agent.online.model.get_weights())
            
            if total_step % MODEL_SAVE_UPDATE_FREQUENCY == 0:
                with open('loss_history.pkl', 'wb') as f:
                    pickle.dump(loss_history, f)
                with open('reward_history.pkl', 'wb') as f:
                    pickle.dump(reward_history, f)

                agent.online.model.save('online_model.h5')

        print('Episode: {}\tTotal reward = {}'.format(current_episode, total_reward))
        reward_history.append(total_reward)
