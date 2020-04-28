"""marl controller."""

import numpy as np
from scipy.spatial.transform import Rotation as R

import gym
import gym.spaces

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
        self.max_timesteps = size * 2 + 6
        self.normalize = normalize
        self.size = size
        self.state = "PICK"

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
        
        x = agent.pos[0] - 2.5 + 0.5
        y = -agent.pos[1] + 2.5 - 0.5
        base_set_pos(x, y)
        
        # reward functions
        if state == "PICK":
            current_distance = self._get_distance(self.agent_pos, self.block_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0
                reward = 10
                self.state = "DROP"
                self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)
                
        if state == "DROP":
            current_distance = self._get_distance(self.agent_pos, self.goal_pos)
            if current_distance < self.prev_distance:
                reward = 1
            elif current_distance > self.prev_distance:
                reward = -1
            elif current_distance == 0
                reward = 10
                self.state = "DONE"
        
            
        self.prev_distance = current_distance
        
        self.timestep += 1
        if self.timestep >= self.max_timesteps or self.state == "DONE":
            done = True
        else:
            done = False
            
        info = {self.state}
                   

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
