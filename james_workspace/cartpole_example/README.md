# reinforcement-learning-warmup
A repo where I conduct custom experiments to get to grips with RL techniques. Most work relating to Cambridge meetups.

## 1. Add the custom cartpole environment 
If you wish to use it, you need to register the custom cartpole environment in gym.

### Copy the environment into the gym package
`cp code/env/custom_cartpole_to_copy.py <path/to/gympackage>/envs/classic_control/custom_cartpole.py`

For example, my command would look like:
`cp code/env/custom_cartpole_to_copy.py ~/anaconda3/envs/my_py36/lib/python3.6/site-packages/gym/envs/classic_control/custom_cartpole.py`

### Add the object to the __init__ file
We then need to make sure gym recognises the object.

Open the following file in your chosen text editor:
`<path/to/gympackage>/envs/classic_control/__init__.py`

And add the line
`from gym.envs.classic_control.custom_cartpole import CustomCartPoleEnv`

This allows the CustomCartPoleEnv to be accessed in the same way as the classic, default CartPole environment.

## 2. Start playing!

Recommend running from root dir `james_workspace` 

### code/main.py
Use this environment to do quick tests with the environments, solving once just to view how your solution is going, for example.

### code/experiments.py
Here I tried to design a few experiments to unpick the behaviour of DQN solver - feel free to repeat or adapt for your own experiments.
