import gym
import numpy as np
from RL_networks import Agent
from custom_2d_env_opencv import *
import cv2
# from utils import plot_learning_curve

render_every = 50

if __name__ == '__main__':
	#env = gym.make('LunarLanderContinuous-v2')
	#env = gym.make('Pendulum-v0')
	# env = gym.make('BipedalWalker-v3')
	video_writer_fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# Creating video writer to save the progress of agent
	video = cv2.VideoWriter('agent_behaviour.avi', video_writer_fourcc, 30, (400, 400))

	env = tbot_2d_env()
	agent = Agent(alpha=0.001, beta=0.001,
			input_dims=env.observation_space.shape, tau=0.005,
			env=env, batch_size=100, layer1_size=400, layer2_size=300,
			n_actions=env.action_space.shape[0])
	n_games = 300
	filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

	best_score = env.reward_range[0]
	score_history = []

	#agent.load_models()

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			if i%render_every == 0:
				img = env.render()
				video.write(img)

			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()

		print('episode ', i, 'score %.1f' % score,
				'average score %.1f' % avg_score)

	# x = [i+1 for i in range(n_games)]

	video.release()
	# plot_learning_curve(x, score_history, filename)