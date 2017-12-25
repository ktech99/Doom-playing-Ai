# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import sys
#sys.path.append('C:/Users/ktech/gym')
import gym
from gym.wrappers import SkipWrapper
#sys.path.append('C:/Users/ktech/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/ktech/.local/lib/python2.7/site-packages')
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete # name of environment

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain

class CNN(nn.Module): #inheriting nn module
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        #convolution is feature detection
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # 1 channel for b&w 3 channels for colour
        # outchanel is number of features, 32 is common practice
        # kernal size is dimensions, usually 2x2, 3x3 or 5x5
        # dimension decreses for accuracy
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) 
        #features increase in last case
        # 3 convolutional layers
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        #hidden layer
        #count neurons > 1=bw 80x80 is dimension 
        # in features is number of pixels being processed
        #out features, higher the better but higher becomes slower
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)#output layer
        #2 full connections to flatten images

    def count_neurons(self, image_dim): #image dimension
        x = Variable(torch.rand(1, *image_dim)) #creating fake image  *is to pass as a list
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) #applying maxpooling, 3 is kernal size, 2 is strides
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))#converting to torch variable
        return x.data.view(1, -1).size(1) #size is to flatten to 1 layer, # data is to access, # view is to view values

    def forward(self, x): #propogate signals in all layers
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1) #flattening layer
        x = F.relu(self.fc1(x))#rectifier function
        x = self.fc2(x)
        return x

# Making the body

class SoftmaxBody(nn.Module): # inherriting from neural network module(not used)
    
    def __init__(self, T): # T is temperature 
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs): #propogating signal from brain to body
        probs = F.softmax(outputs * self.T) #probabilities  
        actions = probs.multinomial() # selecting highest probability
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        #converting images to correct format
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        #converting image to numpy array
        #converting pixels to float 32
        #converting mupy to torch tensor(variable class)
        output = self.brain(input) #passing tensor to brain
        actions = self.body(output) #passing output to body
        return actions.data.numpy() #converting action to numpy



# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True) # importing the environment and preprocessing it
#dimensions should be same as neural network
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True) #saving to videos
number_actions = doom_env.action_space.n # number of actions possible in the environment

# Building an AI
cnn = CNN(number_actions) #calling cnn
softmax_body = SoftmaxBody(T = 1.0) #setting value of temperature for softmax
ai = AI(brain = cnn, body = softmax_body) #calling brain

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) #learning every 10 steps
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000) #creating memory

    
# Implementing Eligibility Trace
#minimizing square distance between input and target while learning
# using algorith in Asynchronus methods for deep Reinforcement learning
def eligibility_trace(batch): #training on batches
    gamma = 0.99 #decay parameter
    inputs = [] #initialising as an empty list
    targets = []  #initialising as an empty list
    for series in batch: #series of 10 transitions
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        #converting first transition and last transition(-1 is a trick to get last)to numpy array
        #converting all values to float 32
        #converting numpy to torch tensor to a tensor variable
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        #cummulative reward
        # 0 if last state of series
        #seies -1 is last value, checking if it's done
        # max of q values if not
        for step in reversed(series[:-1]): # reverse loop, :-1 is up to last element -1
            cumul_reward = step.reward + gamma * cumul_reward #reward of step + gamma value of cummilative reward
        state = series[0].state #firstindex of transition for state
        target = output[0].data #first index of transition for output
        target[series[0].action] = cumul_reward #getting reward for that action of the first state
        # we only take first state as we are taking values for the first step in the batch of 10
        inputs.append(state) #adding first state to series
        targets.append(target) # adding first target to series
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
        #stacking targets and returning them

# Making the moving average on 100 steps
class MA:
    def __init__(self, size): # size is number of steps we want to take an average of
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards): #adding cumilative rewards
        if isinstance(rewards, list): # if rewards are in list format
            self.list_of_rewards += rewards #adding a list
        else:
            self.list_of_rewards.append(rewards) # adding single elements
        while len(self.list_of_rewards) > self.size: #if more than 100 elements
            del self.list_of_rewards[0] #deleting first reward
    def average(self):
        return np.mean(self.list_of_rewards) #computing average of list
ma = MA(100) #average of 100 steps

# Training the AI
loss = nn.MSELoss() #loss function  from nn module
#MSE > mean square error
optimizer = optim.Adam(cnn.parameters(), lr = 0.001) #using adam optimizer
nb_epochs = 100 #size of the number of epochs for trainin the AI
for epoch in range(1, nb_epochs + 1): # from epoch 1 to last epoch
    memory.run_steps(200) #each epoch runs 200 steps
    for batch in memory.sample_batch(128): #batch size as 128(usually 32)
        #every 128 steps memory will give us batches of size 10
        inputs, targets = eligibility_trace(batch) #getting input and target value from eligibility trace
        inputs, targets = Variable(inputs), Variable(targets) # converting to torch variables
        predictions = cnn(inputs) #getting predictions
        loss_error = loss(predictions, targets)# getting loss error between prediction and target
        optimizer.zero_grad() #initialising optimizer
        loss_error.backward() #back propogating loss error
        optimizer.step() #updating weights
    rewards_steps = n_steps.rewards_steps()#commulative rewards of steps
    ma.add(rewards_steps) # adding to moving average
    avg_reward = ma.average() # storing average reward
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward))) #printing the epoch number and avg reward
    if avg_reward >= 1500: #score to reach west
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()
