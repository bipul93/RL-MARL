"""marl controller."""

import numpy as np
from scipy.spatial.transform import Rotation as R

import gym
import gym.spaces

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T
from torch.distributions import Categorical

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Motor, DistanceSensor, Keyboard, Supervisor
# from controller import Robot

# create the Robot instance.
# robot = Robot()
robot = Supervisor()
print(robot)
# print(robot.getField("translation"))
robot_node = robot.getSelf()
print(robot_node.getPosition())
print(robot_node.getOrientation())
r = R.from_rotvec(np.reshape(robot_node.getOrientation(), (3, 3)))
print(r)
print(r.as_euler('zxy'))


trans_field = robot_node.getField("translation")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
# motor = robot.getMotor('motorname')
# ds = robot.getDistanceSensor('dsname')
# print(robot)

# ds.enable(timestep)

# print(robot.getControllerArguments())

wheels = []
SPEED = 4.0

# Get these values from world, (when figured out how to)
floor_size = 5  # 5 x 5 meters
basket_pos = [-2.25, -2.25]
bot_init_pos = [2.0, 2.0]
kuka_box_pos = [2.25, -2.25]

cell_size = 1


def base_set_wheel_speeds_helper(speeds):
    global wheels
    for i in range(4):
        wheels[i].setPosition(float("inf"))
        wheels[i].setVelocity(speeds[i])


def base_init():
    global wheels
    for i in range(4):
        name = "wheel"+str(i+1)
        wheels.append(robot.getMotor(name))
        
def base_get_pos():
    return robot_node.getPosition()
    
def base_set_pos(x, z):
    trans_field.setSFVec3f([x, 0.12, z])
    robot_node.resetPhysics()

def base_reset():
    global wheels
    speeds = [0.0, 0.0, 0.0, 0.0]
    base_set_wheel_speeds_helper(speeds)
    trans_field.setSFVec3f([bot_init_pos[0], 0.12, bot_init_pos[1]])
    robot_node.resetPhysics()
    # robot_node.simulationReset()


def base_forwards():
    global wheels
    speeds = [SPEED, SPEED, SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_backwards():
    global wheels
    speeds = [-SPEED, -SPEED, -SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_turn_left():
    global wheels
    speeds = [-SPEED, SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_turn_right():
    global wheels
    speeds = [SPEED, -SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_left():
    global wheels
    speeds = [SPEED, -SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_right():
    global wheels
    speeds = [-SPEED, SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


class Environment(gym.Env):
    def __init__(self, normalize=False, size=5):
        self.observation_space = gym.spaces.Box(0, size, (size,))
        self.action_space = gym.spaces.Discrete(4)
        self.max_timesteps = size * 2 + 20
        self.normalize = normalize
        self.size = size
        self.state = "PICK"
        base_init()

    def step(self, action):
        action_taken = action
        # ["up", "down", "left", "right"] [0,1,2,3]
        if action_taken == 1:
            self.agent_pos[0] += 1
        elif action_taken == 0:
            self.agent_pos[0] -= 1
        elif action_taken == 3:
            self.agent_pos[1] += 1
        elif action_taken == 2:
            self.agent_pos[1] -= 1
            
        self.agent_pos = np.clip(self.agent_pos, 0, self.size-1)
        
        x = -self.agent_pos[0] + 2.5 - 0.5
        y = self.agent_pos[1] - 2.5 + 0.5
        base_set_pos(x, y)
        
        # reward functions
        reward = 0
        if self.state == "PICK":
            current_distance = self._get_distance(self.agent_pos, self.block_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0:
                reward = 2
                self.state = "DROP"
                self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)
                self.timestep = 0
            else:
                reward = -1
                
        if self.state == "DROP":
            current_distance = self._get_distance(self.agent_pos, self.goal_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0:
                reward = 10
                self.state = "DONE"
            else:
                reward = -1

        self.prev_distance = current_distance
        
        self.timestep += 1
        if self.timestep >= self.max_timesteps or self.state == "DONE":
            done = True
        else:
            done = False
            
        info = {self.state}

        # print(reward)

        return self.agent_pos, reward, done, info
                   

    # distance from kukabox
    def _get_distance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def reset(self):
        self.timestep = 0
        self.agent_pos = [0, 4]
        self.goal_pos = [4, 0]
        self.block_pos = [0, 0]
        self.state = np.zeros((self.size, self.size))
        self.state = "PICK"
        self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)
        #reset bot position
        base_reset()
        return np.array(self.agent_pos) / 1.



env = Environment()
obs = env.reset()

print(env.observation_space)
print(env.action_space.n)

# A2C
class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.linear1 = nn.Linear(2, 128)
    self.dropout = nn.Dropout(p=0.6)
    self.head = nn.Linear(128, env.action_space.n)

  def forward(self, x):
    x = self.linear1(x)
    x = self.dropout(x)
    x = F.relu(x)
    return F.softmax(self.head(x), -1)


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.linear1 = nn.Linear(2, 128)
    self.head = nn.Linear(128, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    return self.head(x.view(x.size(0), -1))



class ACTOR_CRITIC_AGENT():
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-2)
        self.gamma = 0.99
        self.rewards = []
        self.log_probs = []
        self.state_values = []

    # https://pytorch.org/docs/stable/distributions.html
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print(state)
        probs = self.actor(state)
        state_val = self.critic(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, log_prob, state_val  # action, log_prob, state_value

    def truth(self):
        R = 0
        truth = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            truth.insert(0, R)
        truth = torch.tensor(truth)
        truth = (truth - truth.mean())
        return truth

    def optimize_model(self):
        actor_loss = []
        critic_loss = []
        truth = self.truth()
        # print(len(self.log_probs), len(self.rewards))
        for log_prob, value, R in zip(self.log_probs, self.state_values, truth):
            # calculate advantage
            advantage = R - value.item()
            # actor policy loss
            actor_loss.append(-log_prob * advantage)
            # Crtit loss
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor([R]).unsqueeze(1)))

        # Actor loss backprop
        self.actor_optimizer.zero_grad()
        actor_loss = torch.cat(actor_loss).sum()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic loss backprop
        self.critic_optimizer.zero_grad()
        critic_loss = torch.stack(critic_loss).sum()
        critic_loss.backward()
        self.critic_optimizer.step()

        # reset
        del self.rewards[:]
        del self.log_probs[:]
        del self.state_values[:]

    def train(self):
        running_reward = 10
        total_rewards = []
        mean_rewards = []
        # run till it solves
        for i_episode in range(100):  # range(num_episodes):
            state = env.reset()
            # print(state)
            # state = np.reshape(state, [1, 5])

            r = 0
            info_state = ""
            for t in count():
                action, log_prob, state_val = self.select_action(state)
                self.log_probs.append(log_prob)
                self.state_values.append(state_val)
                next_state, reward, done, info = env.step(action.item())
                self.rewards.append(reward)

                # reward = torch.tensor([reward], device=device)
                r += reward  # reward.item()

                # Move to the next state
                state = next_state

                if "DROP" in info:
                    info_state = "DROP"
                if "DONE" in info:
                    info_state = "DONE"
                if done:
                    self.optimize_model()
                    break

            total_rewards.append(r)

            # Exponential moving average
            # running_reward = 0.05 * r + (1 - 0.05) * running_reward
            #
            # # Mean rewards over last 100 episodes
            avg_rewards = np.mean(total_rewards[-100:])
            # mean_rewards.append(avg_rewards)
            # if i_episode % 10 == 0:
            print('Episode {}\tEpisode reward: {:.2f}\tMean-100 episodes: {:.2f}\tinfo_state: {}'.format(i_episode, r, avg_rewards, info_state))
        return total_rewards


agent = ACTOR_CRITIC_AGENT()
total_rewards = agent.train()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    # val = ds.getValue()
    # print(val)
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    # motor.setPosition(10.0)
    base_init()

    # base_forwards()
    
    # base_set_pos(0, 0)
    # base_set_pos(-2.0, -2.0)
    # base_reset()

    pass

# Enter here exit cleanup code.
