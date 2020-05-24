# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.1.0, Keras 2.3.1
# REMARK: The bellow code is modified to run in a kernel from Kaggle (https://www.kaggle.com/)

# %% 
# !pip install gym[atari] # -> uncomment this line if still it's necessary to install gym environment

# %% 

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import gym
import pylab
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import cv2

import tensorflow as tf
import threading
from threading import Thread, Lock
import time

# Used to automate the number of workers
import multiprocessing

from numba import cuda # Release GPU memory

# Remark: It's important use the latest version from tensorflow in this kernel, because using an old version will turn impossible to use 
# the gpu, because there's a specific compatibility with version of both and just is possible downgrade tensorflow, but not cuda
# See this link for more details.: https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

K.clear_session() # reset the graphs and sessions
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True # tells cuda not to use as much VRAM as it wants (as we nneed extra ram for all the other processes)
sess = tf.compat.v1.Session(config=config, graph=tf.compat.v1.get_default_graph())
sess.as_default()

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, activation='linear', kernel_initializer='he_uniform')(X)

    def ppo_loss(y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss =  -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

        return loss
        
    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss=ppo_loss, optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 10000, 0, -21.0 # specific for pong
        self.lock = Lock()
        self.lr = 0.0001

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.EPOCHS = 10
        
        self.keep_running_thread = True # Flag used to kill the threads

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_APPO_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)
        
        self.Actor._make_predict_function()
        self.Actor._make_train_function()
        self.Critic._make_predict_function()
        self.Critic._make_train_function()

        self.session = tf.compat.v1.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()    
        self.graph.finalize()   # graph is not thread-safe, so you need to finilize it... Don't use global graphs with thread
        
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action, prediction

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards, predictions):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        # Compute advantages
        advantages = discounted_r - values

        '''
        pylab.plot(discounted_r,'-')
        pylab.plot(advantages,'.')
        ax=pylab.gca()
        ax.grid(True)
        pylab.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        self.Critic.fit(states, discounted_r, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
 
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow("cartpole"+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, frame, image_memory):
        if image_memory.shape == (1,*self.state_size):
            image_memory = np.squeeze(image_memory)

        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        image_memory = np.roll(image_memory, 1, axis = 0)

        # inserting new frame to free space
        image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(image_memory,0)
        #self.imshow(image_memory,1)
        #self.imshow(image_memory,2)
        #self.imshow(image_memory,3)
        
        return np.expand_dims(image_memory, axis=0)

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame, image_memory)
        return state

    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        next_state = self.GetImage(next_state, image_memory)
        return next_state, reward, done, info
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.reset(self.env)
            done, score, SAVING = False, 0, ''
            # Instantiate or reset games memory
            states, actions, rewards, predictions = [], [], [], []
            while not done:
                #self.env.render()
                # Actor picks an action
                action, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action, self.env, state)
                # Memorize (state, action, reward) for training
                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                predictions.append(prediction)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))

                    self.replay(states, actions, rewards, predictions)
                    
        self.env.close()

    def train(self, n_threads):
        self.env.close()
        # Instantiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self,
                    envs[i],
                    i)) for i in range(n_threads)]

        try: 
            for t in threads:
                time.sleep(2)
                t.start()
            
            for t in threads:
                time.sleep(4)
                t.join()
                
        except (KeyboardInterrupt, SystemExit):
                # Daemon seem doesn't work with kaggle, so all resources allocated
                # by threads will keept used and maybe the own threads don't be finished.
                # That's ocorred when I used kaggle, when I tried finish just the main thread 
                # other threads keept runing even using True Daemon. So a solution was use a flag
                # that switc the states and raise inside the threads turning possible break a lot 
                # of while at once, and the try/except enable doesn't generate a exception error message
                print("########### Exiting all threads...It may take a while ###########")
                self.keep_running_thread = False
                for t in threads:
                    t.join()
                print('All threads are finished....')
                
                # Release resources allocated during training
                self.session.close()
                
                # Release GPU memory - Comment bellow lines if you're not using GPU
                device = cuda.get_current_device()
                device.reset()
            
    def train_threading(self, agent, env, thread):
        try:
            with self.session.as_default():
                while self.episode < self.EPISODES:
                    # Reset episode
                    score, done, SAVING = 0, False, ''
                    state = self.reset(env)
                    # Instantiate or reset games memory
                    states, actions, rewards, predictions = [], [], [], []
                    while not done:
                        # The raise inside the threads turn possible break a lot of while at once,
                        # and the try/except enable doesn't generate a exception error message
                        if not self.keep_running_thread: raise KeyboardInterrupt
                        action, prediction = agent.act(state)
                        next_state, reward, done, _ = self.step(action, env, state)

                        states.append(state)
                        action_onehot = np.zeros([self.action_size])
                        action_onehot[action] = 1
                        actions.append(action_onehot)
                        rewards.append(reward)
                        predictions.append(prediction)

                        score += reward
                        state = next_state

                    self.lock.acquire()
                    self.replay(states, actions, rewards, predictions)
                    self.lock.release()

                    # Update episode count
                    with self.lock:
                        average = self.PlotModel(score, self.episode)
                        # saving best models
                        if average >= self.max_average:
                            self.max_average = average
                            self.save()
                            SAVING = "SAVING"
                        else:
                            SAVING = ""
                        print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, SAVING))
                        if(self.episode < self.EPISODES):
                            self.episode += 1
                env.close()            
        except KeyboardInterrupt:
            print('Thread {} killed.'.format(thread))
        
    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()

# That's a good form for me to automate the choice of number of workers
n_workers = multiprocessing.cpu_count()
print(n_workers)
        
if __name__ == "__main__":
    #env_name = 'PongDeterministic-v4'
    env_name = 'Pong-v0'
    agent = PPOAgent(env_name)
    #agent.run() # use as PPO
    agent.train(n_threads=n_workers) # use as APPO
    #agent.test('Models/Pong-v0_APPO_0.0001_Actor.h5', '')
#     agent.test('Models/Pong-v0_APPO_0.0001_Actor_CNN.h5', '')
