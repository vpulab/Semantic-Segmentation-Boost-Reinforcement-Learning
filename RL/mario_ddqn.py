#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).

## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

from ast import parse
from turtle import back
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from torch.serialization import save
from tqdm import tqdm
import pickle 

from gym_super_mario_bros.actions import RIGHT_ONLY

import gym
import numpy as np
import collections 
from cv2 import cv2
import matplotlib.pyplot as plt
from IPython import display
from segmentator import Segmentator


# Argparse

import argparse
import os
parser = argparse.ArgumentParser()
#Run settings:
parser.add_argument("-vis","--visualization",help="Visualize the game screen",action='store_true')
parser.add_argument("--level",help="What level to play",type=str,default="1-1")
parser.add_argument("--tensorboard",help="Log to tensorboard. Default = True",default=False)
parser.add_argument("--run_name",help="A name for the run. Used in tensorboard. Defaults to Test",type=str,default="Test")
parser.add_argument("-it","--input_type",help="Wether to use semantic segmentation (ss) or normal RGB frames (rgb).",type=str,default="rgb")
parser.add_argument("-inf","--inference_type",help="Wether to run inference with no randomness or maintain a small randomness amount. Can be pure or random",type=str,default='pure')



#Training settings
parser.add_argument("-t","--train",help="Training mode",action='store_true')
parser.add_argument("--max_exp_r",help="Max exploration rate. Defaults to 1", type=float,default=1.0)
parser.add_argument("--min_exp_r",help="Min_exp_rate minimum value for exploration rate",type=float,default=0.02) #if set to 0, it will stop exploring and probably plateau.
parser.add_argument("-e","--epochs",help="Amount of epochs to train for.",type=int,default=2500)
parser.add_argument("-bue","--backup_epochs",help="Backups every e epochs.",type=int,default=0)
parser.add_argument("-sgm","--save_good_model",help="If a model outperforms X times in a row, save it just in case.",type=int,default=-1)

#Model saving and loading
parser.add_argument("-wd","--working_dir",help='Where will files be stored to and loaded from',type=str,default="Models",required=True)
parser.add_argument("-pt","--pretrained_weights",help="Use a pretrained model. Defaults to False",action='store_true')
parser.add_argument("-mn","--model_name",help="Name of the model to load (if different from default)",type=str, default="")


#Other files that can be saved:
parser.add_argument("--load_experience_replay",help="Load a previously saved experience replay dataset. Defaults to false",type=bool,default=False)
parser.add_argument("--save_experience_replay",help="Save the experience replay dataset.Defaults to False. WARNING: Test to 1 or 2 epochs before fully training, or it may give error when saving.",type=bool,default=False)

#parser.add_argument("-h","--help",help="Prints this command and exit",action="store_true")

args = parser.parse_args()

### Run settings.
training = args.train
vis = args.visualization
level = args.level
use_tensorboard = args.tensorboard

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

run_name = args.run_name

if args.input_type == 'ss':
    input_type = 'ss'
    segmentator = Segmentator() #Load segmentation model
else:
    input_type = 'rgb'

##Training settings:
if args.train ==  False:
    if args.inference_type == 'pure':
        max_exploration_rate = 0
        min_exploration_rate = 0
    else:
        max_exploration_rate = args.min_exp_r
        min_exploration_rate = args.min_exp_r
else:
    max_exploration_rate = args.max_exp_r
    min_exploration_rate = args.min_exp_r

epochs = args.epochs

if args.backup_epochs > 0:
    backup_interval = args.backup_epochs
else:
    backup_interval = -1

save_good_model = args.save_good_model
#Model saving and loading
#Is there a directory for models? otherwise create it
dir_exist = os.path.exists(args.working_dir) and os.path.isdir(args.working_dir)
if not dir_exist:
    os.mkdir(args.working_dir)
savepath = args.working_dir+'/'

pretrained_weights = args.pretrained_weights
pretrained_model_name = args.model_name

#What to do with experience replay
load_exp_rep = args.load_experience_replay
save_exp_rep = args.save_experience_replay



##### Setting up Mario environment #########
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    And applies semantic segmentation if set to. Otherwise uses grayscale normal frames.
    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img_og = np.reshape(frame, [240, 256, 3]).astype(np.uint8)
            #If using semantic segmentation:
            if input_type == 'ss':
                img = segmentator.segment_labels(img_og)
                
                #Normalize labels so they are evenly distributed in values between 0 and 255 (instead of being  0,1,2,...)
                img = np.uint8(img*255/6)
            
            else:
                img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        else:
            assert False, "Unknown resolution."

        #Re-scale image to fit model.
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
        x_t = resized_screen[18:102, :]      
        x_t = np.reshape(x_t, [84, 84, 1])

        return x_t.astype(np.uint8)

#Defines a float 32 image with a given shape and shifts color channels to be the first dimension (for pytorch)
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

#Stacks the latests observations along channel dimension
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    #buffer frames. 
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

#Create environment (wrap it in all wrappers)
def make_env(env):
    env = MaxAndSkipEnv(env)
    #print(env.observation_space.shape)
    env = ProcessFrame84(env)
    #print(env.observation_space.shape)

    env = ImageToPyTorch(env)
    #print(env.observation_space.shape)

    env = BufferWrapper(env, 6)
    #print(env.observation_space.shape)

    env = ScaledFloatFrame(env)
    #print(env.observation_space.shape)

    return JoypadSpace(env, RIGHT_ONLY) #Fixes action sets

#### Definition of the DQN model 
class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.gradients = None
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        
        conv_out.register_hook(self.activations_hook)

        return self.fc(conv_out.view(x.size()[0], -1))
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.conv(x)
    
#### Definition of the DQN Agent.
class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay,pretrained):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.local_net = DQNSolver(state_space, action_space).to(self.device)
        self.target_net = DQNSolver(state_space, action_space).to(self.device)
        
        if self.pretrained:
            self.local_net.load_state_dict(torch.load(savepath+pretrained_model_name+"dq1.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(savepath+pretrained_model_name+"dq2.pt", map_location=torch.device(self.device)))
                
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 1000  # Copy the local model weights into the target network every 1000 steps
        self.step = 0

        # Reserve memory for the experience replay "dataset"
        self.max_memory_size = max_memory_size


        if load_exp_rep:
            self.STATE_MEM = torch.load(savepath+"STATE_MEM.pt")
            self.ACTION_MEM = torch.load(savepath+"ACTION_MEM.pt")
            self.REWARD_MEM = torch.load(savepath+"REWARD_MEM.pt")
            self.STATE2_MEM = torch.load(savepath+"STATE2_MEM.pt")
            self.DONE_MEM = torch.load(savepath+"DONE_MEM.pt")

            if True: # If you get errors loading ending positions or num in queue just change this to False
                with open(savepath+"ending_position.pkl", 'rb') as f:
                    self.ending_position = pickle.load(f)
                with open(savepath+"num_in_queue.pkl", 'rb') as f:
                    self.num_in_queue = pickle.load(f)
            else:
                self.ending_position = 0
                self.num_in_queue = 0
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0
        
        self.memory_sample_size = batch_size
        
        #Set up agent learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) #Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done): #Store "remembrance" on experience replay
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        
    def recall(self):
        # Randomly sample 'batch size' experiences from the experience replay
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        # Epsilon-greedy action
        self.step += 1

        if random.random() < self.exploration_rate:  
            return torch.tensor([[random.randrange(self.action_space)]])

        # Local net is used for the policy
        logits = self.local_net(state.to(self.device))
        
        action = torch.argmax(logits).unsqueeze(0).unsqueeze(0).cpu()

        return action

    def copy_model(self):
        # Copy local net weights into target net
        self.target_net.load_state_dict(self.local_net.state_dict())
    
    def experience_replay(self):
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = REWARD + torch.mul((self.gamma * 
                                    self.target_net(STATE2).max(1).values.unsqueeze(1)), 
                                    1 - DONE)
        current = self.local_net(STATE).gather(1, ACTION.long()) # Local net approximation of Q-value
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

#Shows current state (as seen in the emulator, not segmented)
def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)

def run(training_mode, pretrained):
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level
    env = make_env(env)  # Wraps the environment so that frames are grayscale / segmented 
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=4000,
                     batch_size=16,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.,
                     exploration_max=max_exploration_rate,
                     exploration_min=min_exploration_rate,
                     exploration_decay=0.99,
                     pretrained=pretrained)

    #Reset environment
    env.reset()

    #Store rewards and positions
    total_rewards = []
    ending_positions = []
    
    #If using tensorboard initialize summary_writer
    if use_tensorboard == True:
        tensorboard_writer = SummaryWriter('tensorboard/'+run_name+"_labels")

    max_reward = 0
    current_counter = save_good_model

    #Each iteration is an episode (epoch)
    for ep_num in tqdm(range(epochs)):
        #Reset state and convert to tensor
        state = env.reset()
        state = torch.Tensor(np.array([state]))

        #Set episode total reward and steps
        total_reward = 0
        steps = 0
        #Until we reach terminal state
        while True:
            #Visualize or not
            if vis:
                show_state(env, ep_num)
            
            #What action would the agent perform
            action = agent.act(state)
            #Increase step number
            steps += 1
            #Perform the action and advance to the next state
            state_next, reward, terminal, info = env.step(int(action[0]))
            #Update total reward
            total_reward += reward
            #Change to next state
            state_next = torch.Tensor(np.array([state_next]))
            #Change reward type to tensor (to store in ER)
            reward = torch.tensor(np.array([reward])).unsqueeze(0)
            #Get x_position (used only with tensorboard)
            if use_tensorboard == True:
                x_pos = info['x_pos']

            #Is the new state a terminal state?
            terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

            ### Actions performed while training:
            if training_mode:
                #If the episode is finished:
                if terminal:
                    ######################### Model backup section#############################
                    save = False
                    #Backup interval.
                    if ep_num % backup_interval == 0 and ep_num > 0:
                        save = True
                    #Update max reward
                    if max_reward < total_reward:
                        max_reward = total_reward
                    else:
                        #If the model beats a minimum performance level and beats at least a 70% of the max reward it may be a "good" model
                        if (total_reward > 0.7*max_reward and total_reward > 1000):
                            current_counter = current_counter - 1 #reduce counter by one

                            if current_counter == 0: #if the counter reaches 0, model has outperformed X times in a row, save it.
                                save = True
                            elif current_counter < 0: #if the counter is negative, then this saving method is disabled
                                current_counter = -1
                        else:
                            current_counter = save_good_model #if it doesnt, restart coutner.
                    

                    # Save model backup
                    if save == True:
                        with open(savepath+"bp_ending_position.pkl", "wb") as f:
                            pickle.dump(agent.ending_position, f)
                        with open(savepath+"bp_num_in_queue.pkl", "wb") as f:
                            pickle.dump(agent.num_in_queue, f)
                        with open(savepath+run_name+"_bp_total_rewards.pkl", "wb") as f:
                            pickle.dump(total_rewards, f)
                        with open(savepath+run_name+"_bp_ending_positions.pkl", "wb") as f:
                            pickle.dump(ending_positions, f)   

                        torch.save(agent.local_net.state_dict(),savepath+ str(ep_num)+"best_performer_dq1.pt")
                        torch.save(agent.target_net.state_dict(),savepath+ str(ep_num)+"best_performer_dq2.pt")
                            
                        if save_exp_rep: #If save experience replay is on.
                            print("Saving Experience Replay....")
                            torch.save(agent.STATE_MEM,  savepath+"bp_STATE_MEM.pt")
                            torch.save(agent.ACTION_MEM, savepath+"bp_ACTION_MEM.pt")
                            torch.save(agent.REWARD_MEM, savepath+"bp_REWARD_MEM.pt")
                            torch.save(agent.STATE2_MEM,savepath+ "bp_STATE2_MEM.pt")
                            torch.save(agent.DONE_MEM,   savepath+"bp_DONE_MEM.pt")

                ######################### End of Model Backup Section #################################
                #Add state to experience replay "dataset"
                agent.remember(state, action, reward, state_next, terminal)
                #Learn from experience replay.
                agent.experience_replay()

            #Update state to current one
            state = state_next

            #Write to tensorboard Reward and position
            if use_tensorboard and terminal:
                tensorboard_writer.add_scalar('Reward',
                                total_reward ,
                                ep_num)
                tensorboard_writer.add_scalar('Position',
                        x_pos ,
                        ep_num)

            elif terminal == True:
                break #End episode loop

        #Store rewards and positions. Print total reward after episode.
        total_rewards.append(total_reward)
        ending_positions.append(agent.ending_position)
        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
            
    
    if training_mode:
        with open(savepath+"ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(savepath+"num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(savepath+run_name+"_total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        with open(savepath+run_name+"_ending_positions.pkl", "wb") as f:
            pickle.dump(ending_positions, f)
        torch.save(agent.local_net.state_dict(),savepath+ "dq1.pt")
        torch.save(agent.target_net.state_dict(),savepath+ "dq2.pt")


        if save_exp_rep:
            print("Saving Experience Replay....")
            torch.save(agent.STATE_MEM,  savepath+"STATE_MEM.pt")
            torch.save(agent.ACTION_MEM, savepath+"ACTION_MEM.pt")
            torch.save(agent.REWARD_MEM, savepath+"REWARD_MEM.pt")
            torch.save(agent.STATE2_MEM,savepath+ "STATE2_MEM.pt")
            torch.save(agent.DONE_MEM,   savepath+"DONE_MEM.pt")
    else:
        with open(savepath+run_name+"_generalization_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
    
    env.close()
    #Plot rewards evolution
    if training_mode == True:
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot(total_rewards)
        plt.show()


if __name__ == '__main__':
    run(training_mode=training, pretrained=pretrained_weights)

