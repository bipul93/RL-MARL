"""marl controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Motor, DistanceSensor, Keyboard
# from controller import Robot

# create the Robot instance.
robot = Robot()

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


def base_set_wheel_speeds_helper(speeds):
    global wheels
    for i in range(4):
        wheels[i].setPosition(float("inf"))
        wheels[i].setVelocity(speeds[i])


def base_init():
    global wheels
    for i in range(4):
        name = "wheel"+i+1
        wheels.append(robot.getMotor(name))


def base_reset():
    speeds = [0.0, 0.0, 0.0, 0.0]
    base_set_wheel_speeds_helper(speeds)


def base_forwards():
    speeds = [SPEED, SPEED, SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_backwards():
    speeds = [-SPEED, -SPEED, -SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_turn_left():
    speeds = [-SPEED, SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_turn_right():
    speeds = [SPEED, -SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_left():
    speeds = [SPEED, -SPEED, -SPEED, SPEED]
    base_set_wheel_speeds_helper(speeds)


def base_strafe_right():
    speeds = [-SPEED, SPEED, SPEED, -SPEED]
    base_set_wheel_speeds_helper(speeds)



# m1 = robot.getMotor("wheel1")
# m2 = robot.getMotor("wheel2")
# m3 = robot.getMotor("wheel3")
# m4 = robot.getMotor("wheel4")
# print(m1.getVelocity())
#
# m1.setPosition(float("inf"))
# m1.setVelocity(4.0)
#
# m2.setPosition(float("inf"))
# m2.setVelocity(4.0)
#
#
# m3.setPosition(float("inf"))
# m3.setVelocity(4.0)
#
#
# m4.setPosition(float("inf"))
# m4.setVelocity(4.0)



 

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) == 1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    # val = ds.getValue()
    # print(val)
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    # motor.setPosition(10.0)
    base_init()

    pass

# Enter here exit cleanup code.
