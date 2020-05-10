"""marl controller."""
import math

print("Test")

import numpy as np
from scipy.spatial.transform import Rotation as R

import gym
import gym.spaces

from collections import namedtuple
from itertools import count
from PIL import Image

import time

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
# print(robot.getField("translation").getSFVec3f())
# print(robot.getField("rotation"))
robot_node = robot.getSelf()

box1 = robot.getFromDef("box1")
box2 = robot.getFromDef("box2")
print(box1.getPosition())
print(robot_node.getPosition())
print(robot_node.getOrientation())
print((robot_node.getField("rotation")).getSFRotation())
r = R.from_rotvec(np.reshape(robot_node.getOrientation(), (3, 3)))
print(r)
print(r.as_euler('zxy'))

trans_field = robot_node.getField("translation")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print(timestep)
print(robot.getTime())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
# motor = robot.getMotor('motorname')
# ds = robot.getDistanceSensor('dsname')
# print(robot)

# ds.enable(timestep)

# print(robot.getControllerArguments())

m = (robot_node.getField("translation")).getSFVec3f()
x = round(-m[0] + 2.5 - 0.5)
y = round(m[2] + 2.5 - 0.5)
print("->", m, x, y)

wheels = []
SPEED = 14.0

# Get these values from world, (when figured out how to)
floor_size = 5  # 5 x 5 meters
basket_pos = [-2.25, -2.25]
bot_init_pos = [x, y]
kuka_boxes = 2
kuka_box_pos = [[2.25, -2.25], [-2.25, 2.25]]

boxes = [
    {
        "pos": [2.25, -2.25],
        "grid_pos": [0, 0],
        "node": box1
    },
    {
        "pos": [2.25, -2.25],
        "grid_pos": [4, 4],
        "node": box2
    }

]

cell_size = 1


def base_set_wheel_speeds_helper(speeds):
    global wheels
    for i in range(4):
        wheels[i].setPosition(float("inf"))
        wheels[i].setVelocity(speeds[i])


def base_init():
    global wheels
    for i in range(4):
        name = "wheel" + str(i + 1)
        wheels.append(robot.getMotor(name))


def base_get_pos():
    return robot_node.getPosition()


def base_set_pos(x, z, reset):
    trans_field.setSFVec3f([x, 0.12, z])
    if reset:
        robot_node.resetPhysics()


def base_reset():
    global wheels
    speeds = [0.0, 0.0, 0.0, 0.0]
    base_set_wheel_speeds_helper(speeds)
    # robot_node.resetPhysics()
    # trans_field.setSFVec3f([bot_init_pos[0], 0.12, bot_init_pos[1]])
    # robot_node.resetPhysics()
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
    speeds = [SPEED, -SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_turn_right():
    global wheels
    speeds = [-SPEED, SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_left():
    global wheels
    speeds = [SPEED, -SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_right():
    global wheels
    speeds = [-SPEED, SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def get_base_rotation():
    return (robot_node.getField("rotation")).getSFRotation()


def get_base_position():
    return (robot_node.getField("translation")).getSFVec3f()


class Environment(gym.Env):
    def __init__(self, normalize=False, size=5):
        self.observation_space = gym.spaces.Box(0, size, (size,))
        self.action_space = gym.spaces.Discrete(4)
        self.bot_state_space = gym.spaces.Discrete(2)
        self.max_timesteps = size * size * 100
        self.normalize = normalize
        self.size = size
        self.bot_state = 0  # 0 or 1 or 2 - PICK or DROP or DONE
        base_init()
        self.boxes = boxes.copy()

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

        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # x = -self.agent_pos[0] + 2.5 - 0.5
        # y = self.agent_pos[1] - 2.5 + 0.5
        # base_set_pos(x, y)

        # reward functions
        reward = 0
        if self.bot_state == 0:
            current_distance = self._get_distance(self.agent_pos, self.block_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0:
                reward = 2
                self.bot_state = 1
                self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)
                self.timestep = 0
                # print("Agent picked box 0")
            else:
                reward = -1

        if self.bot_state == 1:
            current_distance = self._get_distance(self.agent_pos, self.goal_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0:
                # drop and check for next box
                self.boxes.pop(0)
                if len(self.boxes) > 0:
                    # print("Agent dropped box 0")
                    reward = 1
                    self.bot_state = 0
                    self.block_pos = self.get_next_box_pos()
                    self.prev_distance = self._get_distance(self.agent_pos, self.block_pos)
                    self.timestep = 0
                    # print("Agent need to pick box 1", self.block_pos, self.bot_state)
                else:
                    reward = 10
                    self.bot_state = 2
                    # print("Agent dropped box 1")
            else:
                reward = -1

        self.prev_distance = current_distance

        self.timestep += 1
        if self.timestep >= self.max_timesteps or self.bot_state == 2:
            # if self.bot_state == 2:
            done = True
        else:
            done = False

        info = {self.bot_state}

        # print(reward)

        obs = np.array([self.agent_pos[0], self.agent_pos[1], self.bot_state, len(self.boxes)]) / 1

        return obs, reward, done, info

    # distance from kukabox
    def _get_distance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def get_next_box_pos(self):
        if len(self.boxes) > 0:
            return self.boxes[0].get("grid_pos")
        else:
            return []

    def reset(self):
        self.timestep = 0
        self.agent_pos = [bot_init_pos[0], bot_init_pos[1]]
        self.goal_pos = [4, 0]
        self.boxes = boxes.copy()
        self.block_pos = self.get_next_box_pos()
        # print("Reset", self.boxes, self.block_pos)
        self.bot_state = 0
        self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)
        # reset bot position
        # base_reset()
        # print(self.agent_pos[0], self.agent_pos[1], self.bot_state)
        # print(np.array(self.agent_pos) / 1)
        return np.array([self.agent_pos[0], self.agent_pos[1], self.bot_state, len(self.boxes)]) / 1

    def render(self, action):
        x = -self.agent_pos[0] + 2.5 - 0.5
        y = self.agent_pos[1] - 2.5 + 0.5
        bot_pos = get_base_position()
        dist = math.sqrt(((x - bot_pos[0]) ** 2) + ((y - bot_pos[2]) ** 2))
        print("Agent pos: ", x, y, action, bot_pos, dist)
        if action == -1:
            return
        if action == 2:
            print("action2")
            # passive_wait(5)
            # base_turn_left()
            # # passive_wait(10)
            # time.sleep(10)
            # # if (abs(-2.09 - get_base_rotation()[3])) >= 0.01:
            # #     print (abs(-2.09 - get_base_rotation()[3]))
            # #     break;
            # base_reset()
            # base_forwards()
            # passive_wait(5)
            # if dist >= 0.1:
            #     base_forwards()
            #     passive_wait(5)
            # base_reset()
        # else:
        # while True:
        # base_turn_left()
        base_set_pos(x, y, True)
        passive_wait(5.0)
        time.sleep(2)
        return


env = Environment()
obs = env.reset()

print(env.observation_space)
print(env.action_space)
print(obs)


# A2C
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        # self.dropout = nn.Dropout(p=0.6)
        self.head = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        return F.softmax(self.head(x), -1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.head(x.view(x.size(0), -1))


class ACTOR_CRITIC_AGENT():
    def __init__(self):
        # self.actor = Actor()
        self.actor = torch.load("model2.pth")
        self.actor.eval()
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
        DONE_COUNT = 0
        for i_episode in count():  # range(num_episodes):
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

                if 1 in info:
                    info_state = "DROP"
                elif 2 in info:
                    info_state = "DONE"
                else:
                    info_state = info
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
            print('Episode {}\tEpisode reward: {:.2f}\tMean-100 episodes: {:.2f}\tinfo_state: {}'.format(i_episode, r,
                                                                                                         avg_rewards,
                                                                                                         info_state))
            if info_state == "DONE":
                DONE_COUNT += 1
            else:
                DONE_COUNT = 0

            if DONE_COUNT == 20:
                break

        return total_rewards

    def save_model(self):
        torch.save(self.actor, "model2.pth")
        return

    def evaluate(self):
        obs = env.reset()
        done = False
        env.render(-1)
        print("--> ", obs)
        while not done:
            action, log_prob, state_val = self.select_action(obs)
            obs, reward, done, info = env.step(action.item())
            print(action.item(), info, done)
            env.render(action.item())


agent = ACTOR_CRITIC_AGENT()

# Remove model load from actor network if training
# Change model name before saving
# total_rewards = agent.train()
# agent.save_model()


def passive_wait(sec):
    start_time = robot.getTime()
    while True:
        step()
        if start_time + sec > robot.getTime():
            break


def step():
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    if robot.step(timestep) == -1:
        return


complete = True
moveToNextState = True
obs = env.reset()
done = False
# obs, reward, done, info
print("evaluating ... ")
while complete:
    step()

    # Dont remove following lines, Its for jump testing
    # print("evaluating ... ")
    # agent.evaluate()
    # complete = False
    #
    if moveToNextState and not done:
        moveToNextState = False
        action, log_prob, state_val = agent.select_action(obs)
        obs, reward, done, info = env.step(action.item())
    # env.render(action.item())
    # print(obs, "-->", get_base_rotation())
    x = -obs[0] + 2.5 - 0.5
    y = obs[1] - 2.5 + 0.5
    bot_pos = get_base_position()
    dist = math.sqrt(((x - bot_pos[0]) ** 2) + ((y - bot_pos[2]) ** 2))
    # print(x, y, obs, bot_pos, dist)

    within_cell_bounds = False
    # print("within", obs, [x, y], abs(bot_pos[0] - x),  abs(bot_pos[2] - y))
    if abs(bot_pos[0] - x) <= 0.3 and abs(bot_pos[2] - y) < 0.3:
        within_cell_bounds = True
        # print("TRUE")
        # base_reset()
        # break


    if action == 2:  # left
        angle_val = -2.09
    if action == 1:  # down
        angle_val = -3.14
    if action == 3:  # right
        angle_val = -2.09
    if action == 0:  # up
        angle_val = -1.57

    # ToDo: Optimize the below conditions
    if (abs(-1.57 - get_base_rotation()[3])) <= 0.01: # Heading up
        heading = "UP"
        if action == 2:  # left
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_left()
        if action == 1:  # down
            base_backwards()
        if action == 3:  # right
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_right()
        if action == 0:  # up
            base_forwards()

    elif (abs(-3.14 - get_base_rotation()[3])) <= 0.01: # Heading Down
        heading = "DOWN"
        if action == 2:  # left
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_right()
        if action == 1:  # down
            base_forwards()
        if action == 3:  # right
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_left()
        if action == 0:  # up
            base_backwards()

    elif (abs(-2.09 - get_base_rotation()[3])) <= 0.01 and get_base_rotation()[1] < 0 and get_base_rotation()[2] < 0: # Heading Left
        heading = "LEFT"
        if action == 2:  # left
            base_forwards()
        if action == 1:  # down
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_left()
        if action == 3:  # right
            base_backwards()
        if action == 0:  # up
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_right()

    elif (abs(-2.09 - get_base_rotation()[3])) <= 0.01 and get_base_rotation()[1] > 0 and get_base_rotation()[2] > 0: # Heading Right
        heading = "RIGHT"
        if action == 2:  # left
            base_backwards()
            # or
            # base_turn_left()
        if action == 1:  # down
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_right()
        if action == 3:  # right
            base_forwards()
        if action == 0:  # up
            if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
                base_turn_left()
    # else:
    #     # print("ERROR: ", "Unknown heading direction")
    #     if (abs(angle_val - get_base_rotation()[3])) <= 0.01:
    #         print("same angle base reset")
    #         base_reset()
    # if (abs(angle_val - get_base_rotation()[3])) >= 0.01:
    #     base_turn_left()
    # else:
    #     base_reset()
    #     base_forwards()

    print("Agent pos: ", obs, x, y, action, bot_pos, dist, done, angle_val, abs(angle_val - get_base_rotation()[3]), get_base_rotation()[3], within_cell_bounds, heading)

    # incorporating physics
    if dist <= 0.15:
        base_reset()
        # if obs[2] == 1:
        #     print("PICKING....")
        #     for i in boxes:
        #         if abs(bot_pos[0] - i["pos"][0]) <= 0.5 and abs(bot_pos[2] - i["pos"][1]) < 0.5:
        #             box_node = i["node"]
        #             if heading == "UP":
        #                 pos = [bot_pos[0] - 0.15, 0.19, bot_pos[2]]
        #             elif heading == "LEFT":
        #                 pos = [bot_pos[0], 0.19, bot_pos[2]+0.20]
        #             elif heading == "DOWN":
        #                 pos = [bot_pos[0], 0.19, bot_pos[2]+0.20]
        #             elif heading == "RIGHT":
        #                 pos = [bot_pos[0], 0.19, bot_pos[2]-0.20]
        #             print(box_node.getField("translation").getSFVec3f(), bot_pos, heading, pos)
        #             box_node.getField("translation").setSFVec3f(pos)
        #             # complete = False
        #             break
        # time.sleep(1)
        moveToNextState = True
        # passive_wait(0.2)
    # elif within_cell_bounds:
    #     # setting position explicitly
    #     base_reset()
    #     base_set_pos(x, y, True)
    #     moveToNextState = True
    #     print("WITHIN cell")

    if done:
        complete = False
