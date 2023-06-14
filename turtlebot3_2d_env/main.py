import gym
import numpy as np
from custom_2d_env_opencv import *
from RL_networks import Agent
# import cv2

render_every = 30

if __name__ == '__main__':
	video_writer_fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# Creating video writer to save the progress of agent
	video = cv2.VideoWriter('agent_behaviour.avi', video_writer_fourcc, 30, (400, 400))

	env = tbot_2d_env()
	agent = Agent(agent_lr=0.0005, critic_lr=0.001,
			input_dims=env.observation_space.shape, tau=0.005,
			env=env, batch_size=100, layer1_size=512, layer2_size=256, layer3_size=128,
			n_actions=env.action_space.shape[0])
	# agent.load_models()
	n_eps = 2000

	best_score = env.reward_range[0]
	score_history = []

	#agent.load_models()

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			# print("Action is: ", action)
			observation_, reward, done, info = env.step(action)
			# if i%render_every == 0:
				# img = env.render()
				# video.write(img)

			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()

		# if agent.ep_greedy < 0.995 and (i%20) == 0:
		# 	agent.ep_greedy += 0.005

		print('episode', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

	video.release()
	avg_rn = np.zeros(len(score_history))
	last_n_eps = 100
	x_axis = [j+1 for j in range(len(score_history))]
	for i in range(len(avg_rn)):
		avg_rn[i] = np.mean(score_history[max(0, i-last_n_eps):(i+1)])
	plt.plot(x_axis, score_history)
	# plt.title('Plot of running average of previous {} scores'.format(last_n_eps))
	plt.title('Plot of scores w.r.t. episodes for {} episodes'.format(len(score_history)))
	# plt.savefig(filename)
	plt.show()