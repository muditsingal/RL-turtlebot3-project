import cv2
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
import random
import time

dt = 0.1

class tbot_2d_env(Env):
	global dt
	def get_obs_tuple(x, y, r):
		return []
	def __init__(self):
		self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1])) # Actions v, w ranging from -1 to 1 (scaled later) 
		# Observation space of env, 12 lidar rays, angle and distance of agent to traget, linear and angular velocities of prev step 
		# self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0, -np.pi, 0, -1, -1]), high=np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, np.pi, 5.7, 1, 1]))
		# Observation space of env, 12 lidar rays, x, y, theta pos of agent, x, y, theta pos of goal linear and angular velocities of prev step 
		self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0, 0.0, 0.0, -np.pi, 0.0, 0.0, -np.pi, -1, -1]), 
									 high=np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, 4.0, 4.0, np.pi, 4.0, 4.0, np.pi,  1, 1]))
		# Initializing the robot's starting state, random lidar data, 12 lidar rays, x, y, theta pos of agent, x, y, theta pos of goal, linear and angular velocities of initial step
		self.state = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, 0.3, 0.3, 0, 3.3, 3.3, 0, 0, 0])
		self.ep_duration = 200
		self.obstacles = [np.array([1.5, 1.5, 0.2]), np.array([2.5, 1.5, 0.2]), np.array([2.5, 2.5, 0.2]), np.array([1.5, 2.5, 0.2])]
		self.img = np.ones((400,400,3), dtype=np.uint8)*255

	def random_start(self):
		self.state[12] = np.random.random() + 0.3
		self.state[13] = np.random.random() + 0.3
		self.state[14] = 0

	def random_goal(self):
		self.state[15] = np.random.random() + 2.8
		self.state[16] = np.random.random() + 2.8
		self.state[17] = 0

	def reset(self):
		self.state = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, 0.3, 0.3, 0, 3.3, 3.3, 0, 0, 0])
		self.random_start()
		self.random_goal()
		# self.state[15] = 0.8
		# self.state[16] = 1.1
		# self.state[17] = 0
		self.ep_duration = 200
		return self.state

	def step(self, action):
		r_i = 0.1 	# Radius of robot
		r_s = 0.4 	# Safety radius from obstacles
		reward = 0.0
		alpha = 0.5
		d_offset = 5
		done = False
		# blank info required by openAI gym
		info = {}

		d_to_goal = np.linalg.norm(self.state[12:14] - self.state[15:17])
		self.ep_duration -= 1
		# update state by performing action
		v = action[0]
		w = action[1]
		dt = 0.1

		# Updating the angle and distance of agent to target
		self.state[18] = v  	# Velocity in state
		self.state[19] = w  	# Angular velocity in state

		th = self.state[14]

		for i in range(10):	
			self.state[12] += v*dt*np.cos(th)
			self.state[13] += v*dt*np.sin(th)
			th += w*dt

		self.state[14] = th
		# Read lidar data from new position after perfroming action
		# self.state [0:12] = get_lidar_data(self.state[12], self.state[13])
		self.get_lidar_data()

		# calculate reward
		# Checking distance from the goal position
		if d_to_goal < 0.15:
			reward += 200
			done = True
			return self.state, reward, done, info

		# Checking if robot moves out of the arena (4mx4m) area, giving a large negative reward and marking done if robot goes out of arena
		if self.state[12] < r_i or self.state[13] < r_i or self.state[12] > 400 - r_i or self.state[13] > 400 - r_i:
			reward -= 150
			done = True
			return self.state, reward, done, info

		
		for ray_scan in self.state[0:12]:
			if ray_scan < 0.4:
				reward -= 150
				done = True
				break

			elif ray_scan < (r_i + r_s) :
				reward += (ray_scan - 1.5)/(r_i + r_s)

			else:
				reward += alpha*(d_offset - d_to_goal)

		# if ep_duration <= 0, mark done
		if self.ep_duration <= 0 or done:
			done = True

		else:
			done = False

		# return updated state, reward, done flag, info
		return self.state, reward, done, info


	def get_lidar_data(self):
		self.state[0:12] = 1.5
		for obs in self.obstacles:
			d = np.linalg.norm(self.state[12:14] - obs[0:2]) 
			theta_rbt = self.state[14]
			rbt_x = self.state[12]
			rbt_y = self.state[13]
			if d > 1.5:
				continue

			theta = (np.arctan2(obs[1]-self.state[13], obs[0]-self.state[12]) - self.state[14])*180/np.pi
			if theta < 0:
				theta = 360 + theta

			theta = theta%360
			ray_id = int(theta//30)
			
			for i in range(3):
				b = 1
				m = (np.tan(theta_rbt + 30*(ray_id + i - 1)*np.pi/180))
				a = -m
				c = m*rbt_x - rbt_y
				p_dist = (np.abs(a*rbt_x + b*(rbt_y) + c))/np.sqrt(a**2 + b**2)
				if p_dist < obs[2]:
					# print(ray_id)
					self.state[(ray_id + i - 1)%12] = d - obs[2]


		# Use current robot position and orientation, obstacle position and shape to get the intersection point of ray with the obstacle

	def render(self):
		self.img = np.ones((400,400,3), dtype=np.uint8)*255
		cv2.circle(self.img, (int(self.state[12]*100), int(400-self.state[13]*100)), int(0.1*100), (255, 0, 0), thickness=-1)
		cv2.circle(self.img, (int(self.state[15]*100), int(400-self.state[16]*100)), int(0.1*100), (0, 255, 0), thickness=-1)
		for obs in self.obstacles:
			cv2.circle(self.img, (int(obs[0]*100), int(400-obs[1]*100)), int(obs[2]*100), (120, 120, 120), thickness=-1)

		return self.img
		# cv2.imshow("Enviroment window", self.img)



if __name__=='__main__':

	env = tbot_2d_env()
	print(env.action_space.sample())
	rand_state = env.reset()


	exit = False

	ep_num = 1
	step_num = 0
	print(env.action_space.low[1])


	while ep_num < 22:
		step_num = 0
		env.reset()
		while step_num < 202:
			step_num += 1
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			# if done == True and reward > 20:
			# 	print(state[:12], done, reward)

			if ep_num %100 == 0:
				action = env.action_space.sample()
				# action = np.array([0.1, 0.05])
				# state, reward, done, info
				state, reward, done, info = env.step(action)
				if done == True and reward > 20:
					print(state[:12], done, reward)

				env.render()
				cv2.waitKey(1)

		ep_num += 1

	cv2.destroyAllWindows()
