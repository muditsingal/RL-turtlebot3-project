# RL-turtlebot3-project
Repo for a project for using reinforcement learning on the turtlebot3

## Libraries:
Tensorflow, OpenCV, os, time

## General information:
In this project I have created a reinforcement learning environment from scratch. It inherits the Environment class from OpenAI's gym library, to allow easy usage of RL techniques from gym environments. This also allows the environment to be easily plugged into existing models.

This project uses **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm, which is an actor-critic based Deep RL technique. The main advantage of this algorithm is that it helps in avoiding over-estimation bias that comes with any DNN based technique.

The render function is also written from scratch which helps in observing the progress made by the agent during training.

I will be updating this repository in the coming days with detailed implementation and results!

