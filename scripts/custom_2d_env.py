import pygame
import cv2
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
import random
import time

img = np.ones([480, 640, 3], dtype=np.uint8)*255
cv2.imshow("Blank", img)
cv2.waitKey(1)
cv2.destroyAllWindows()


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
		# Initializing the robot's starting state, random lidar data, angle and distance difference from target, v and w
		self.state = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, 0.3, 0.3, 0, 3.3, 3.3, 0, 0, 0])
		self.ep_duration = 200
		self.obstacles = [np.array([1.5, 1.5, 0.2]), np.array([2.5, 1.5, 0.2]), np.array([2.5, 2.5, 0.2]), np.array([1.5, 2.5, 0.2])]
		self.window = pygame.display.set_mode((400, 400))	# Width, height of window

	def random_start(self):
		self.state[12] = np.random.random() + 0.2
		self.state[13] = np.random.random() + 0.2
		self.state[14] = 0

	def random_goal(self):
		self.state[15] = np.random.random() + 3
		self.state[16] = np.random.random() + 3
		self.state[17] = 0

	def reset(self):
		state = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5, 0.3, 0.3, 0, 3.3, 3.3, 0, 0, 0])
		self.random_start()
		self.random_goal()
		self.ep_duration = 200
		return self.state

	def step(self, action):
		self.ep_duration -= 1
		# update state by performing action
		v = action[0]
		w = action[1]
		dt = 0.1

		# Updating the angle and distance of agent to target
		self.state[18] = v  	# Velocity in state
		self.state[19] = w  	# Angular velocity in state


		for i in range(10):
			th = self.state[14]
			self.state[12] += v*dt*np.cos(th)
			self.state[13] += v*dt*np.sin(th)
			th += w*dt

		# Read lidar data from new position after perfroming action
		# self.state [0:12] = get_lidar_data(self.state[12], self.state[13])
		get_lidar_data()

		# calculate reward
		# reward = 


		# if ep_duration <= 0, mark done
		if self.ep_duration <= 0:
			done = True

		else:
			done = False

		# blank info required by openAI gym
		info = {}
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
			ray_id = theta//30
			
			for i in range(3):
				b = 1
				m = (np.tan(theta_rbt + 30*(ray_id + i - 1)*np.pi/180))
				a = -m
				c = m*rbt_x - rbt_y
				p_dist = (np.abs(a*rbt_x + b(rbt_y) + c))/np.sqrt(a**2 + b**2)
				if p_dist < obs[2]:
					self.state[ray_id + i - 1] = d - obs[2]


		# Use current robot position and orientation, obstacle position and shape to get the intersection point of ray with the obstacle

	def render(self):
		pygame.display.set_caption("Demo window")
		self.window.fill((255, 255, 255))
		pygame.draw.circle(self.window, (0, 0, 255), (self.state[12]*100, 400-self.state[13]*100), 0.1*100)
		pygame.draw.circle(self.window, (0, 255, 0), (self.state[15]*100, 400-self.state[16]*100), 0.1*100)
		for obs in self.obstacles:
			pygame.draw.circle(self.window, (120, 120, 120), (obs[0]*100, 400-obs[1]*100), obs[2]*100)

		pygame.display.flip()
		pygame.display.update()


	# def render(self, episode, render_every):
	# 	exit = False
	# 	while not exit:
	# 		for event in pygame.event.get():
	# 			if event.type == pygame.QUIT:
	# 				# pygame.quit()
	# 				exit = True
	# 				break

	# 			pygame.display.set_caption("Demo window")
	# 			self.window.fill((255, 255, 255))
	# 			pygame.draw.circle(self.window, (0, 0, 255), (self.state[12]*100, 400-self.state[13]*100), 0.1*100)
	# 			pygame.draw.circle(self.window, (0, 255, 0), (self.state[15]*100, 400-self.state[16]*100), 0.1*100)
	# 			for obs in self.obstacles:
	# 				pygame.draw.circle(self.window, (120, 120, 120), (obs[0]*100, 400-obs[1]*100), obs[2]*100)

	# 			pygame.display.flip()
	# 			pygame.display.update()





pygame.init()
env = tbot_2d_env()
print(env.action_space.sample())
rand_state = env.reset()

# env.render(1, 1)


exit = False

ep_num = 1
step_num = 0

while not exit:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			# pygame.quit()
			exit = True
			break

	env.render()


# while ep_num < 1000:
# 	step_num = 0
# 	while step_num < 100:
# 		if ep_num %5 == 0:
# 			env.render()

# 		step_num += 1
# 		# time.sleep(0.001)

# 	ep_num += 1



'''
window = pygame.display.set_mode((400, 400))	# Width, height of window
pygame.display.set_caption("Demo window")

exit = False

x, y = window.get_size()
print("Size of game window is: {}px in height and {}px in width".format(y,x))

while not exit:
	window.fill((255, 255, 254))
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			# pygame.quit()
			exit = True
			break

	pygame.draw.rect(window, (0, 255, 0), pygame.Rect(25, 25, 75, 75))
	pygame.draw.circle(window, (0, 0, 255), (350, 200), 80)
	pygame.display.flip()
	pygame.display.update()

'''