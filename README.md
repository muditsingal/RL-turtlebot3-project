# RL-turtlebot3-project
Repo for a project for using reinforcement learning on the turtlebot3


## General information:
In this project I have created a reinforcement learning environment from scratch. It inherits the Environment class from OpenAI's gym library, to allow easy usage of RL techniques from gym environments. This also allows the environment to be easily plugged into existing models.

This project uses **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm, which is an actor-critic based Deep RL technique. The main advantage of this algorithm is that it helps in avoiding over-estimation bias that comes with any DNN based technique.

The render function is also written from scratch which helps in observing the progress made by the agent during training.

## Libraries
OpenCV, pyGame, Numpy, Tensorflow, Pytorch, OpenAI gym, 
math, subprocess, time, squaternion, Tensorboard, random, collections
## The steps to run for each part are mentioned below:
### 1. Turtlebot 3 in 2D environment (Files located in Folder: turtlebot3_2d_env)
1. Go to the folder turtlebot3_2d_env and open the main.py file.
2. Change the render_every file to change how often episodes are captured
3. Change the n_eps parameter to set the maximum number of episodes the agent should be trained on
4. You can also change the agent_lr, critic_lr, and tau paraameters which are the parameters that control the learning rates of the actor, critic networks and tau contorls the rate at which the target networks are updated w.r.t. actor and critic networks.
5. Run the scripts until training concludes to get the plot and video recording of agent's performance.

You can play around in the 2D environment by running the teleop_2d_env_opencv.py file.


### 2. Bipedal walker in 2D environment with TD3 (Files located in Folder: bipedal_walker_2D)
1. Go to the folder bipedal_walker_2D_env and open the TD3_walker.ipynb file.
2. Change the render_every file to change how often episodes are captured
3. Change the n_eps parameter to set the maximum number of episodes the agent should be trained on
4. You can also change the agent_lr, critic_lr, and tau paraameters which are the parameters that control the learning rates of the actor, critic networks and tau contorls the rate at which the target networks are updated w.r.t. actor and critic networks.
5. Run the scripts until training concludes to get the plot and video recording of agent's performance.

