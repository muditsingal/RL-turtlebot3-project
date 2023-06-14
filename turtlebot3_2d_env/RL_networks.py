# Importing all necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

# The replay buffer class
class replay_buffer:
	# Initialize the memory buffer. The size represents the number of individual experiences, 
	# mem_cntr keeps track of the replay index and prevents out-of-index errors
	# state mem stores the observations of current time step
	# new state mem stores the observations of next time step after action a has been performed (the transition)
	# The action and reward mem store the action and reward corresponding to that particular time step 
	# Terminal mem stores the if a particular time step was the terminal step (i.e. if the episode was marked done at that step)
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.state_mem = np.zeros((self.mem_size, *input_shape))
		self.new_state_mem = np.zeros((self.mem_size, *input_shape))
		self.action_mem = np.zeros((self.mem_size, n_actions))
		self.reward_mem = np.zeros(self.mem_size)
		self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool_)


	def store_in_buffer(self, state, action, reward, state_new, done):
		idx = self.mem_cntr % self.mem_size

		self.state_mem[idx] = state
		self.new_state_mem[idx] = state_new
		self.action_mem[idx] = action
		self.reward_mem[idx] = reward
		self.terminal_mem[idx] = done

		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)

		batch = np.random.choice(max_mem, batch_size)

		states = self.state_mem[batch]
		new_states = self.new_state_mem[batch]
		actions = self.action_mem[batch]
		rewards = self.reward_mem[batch]
		dones = self.terminal_mem[batch]

		return states, actions, rewards, new_states, dones


class actor_nw(keras.Model):
	def __init__(self, fc1_dims, fc2_dims, fc3_dims, n_actions, name, chkpt_dir='tmp/td3'):
		super(actor_nw, self).__init__()
		self.layer1_dims = fc1_dims
		self.layer2_dims = fc2_dims
		self.layer3_dims = fc3_dims
		self.n_actions = n_actions
		self.model_name = name
		self.check_pt_dir = chkpt_dir
		self.check_pt_file = os.path.join(self.check_pt_dir, name+'_td3')

		self.layer1 = Dense(self.layer1_dims, activation='relu')
		self.layer2 = Dense(self.layer2_dims, activation='relu')
		self.layer3 = Dense(self.layer3_dims, activation='relu')
		self.mu = Dense(self.n_actions, activation='tanh')
	
	def call(self, state):
		actor_op = self.layer1(state)
		actor_op = self.layer2(actor_op)
		actor_op = self.layer3(actor_op)

		mu = self.mu(actor_op)

		return mu

class critic_nw(keras.Model):
	def __init__(self, fc1_dims, fc2_dims, fc3_dims, name, chkpt_dir='tmp/td3'):
		super(critic_nw, self).__init__()
		self.layer1_dims = fc1_dims
		self.layer2_dims = fc2_dims
		self.layer3_dims = fc3_dims
		self.model_name = name
		self.check_pt_dir = chkpt_dir
		self.check_pt_file = os.path.join(self.check_pt_dir, name+'_td3')

		self.layer1 = Dense(self.layer1_dims, activation='relu')
		self.layer2 = Dense(self.layer2_dims, activation='relu')
		self.layer3 = Dense(self.layer3_dims, activation='relu')
		self.q_op = Dense(1, activation=None)

	def call(self, state, action):
		q_value = self.layer1(tf.concat([state, action], axis=1))
		q_value = self.layer2(q_value)
		q_value = self.layer3(q_value)

		q = self.q_op(q_value)

		return q

# The agent class
class Agent:
	def __init__(self, agent_lr, critic_lr, input_dims, tau, env,
				 gamma=0.99, update_actor_interval=2, warmup=100,
				 n_actions=2, max_size=1000000, layer1_size=512,
				 layer2_size=256, layer3_size=128, batch_size=100, noise=0.1):
		self.gamma = gamma
		self.tau = tau
		self.max_action = env.action_space.high[0]
		self.min_action = env.action_space.low[0]
		self.mem = replay_buffer(max_size, input_dims, n_actions)
		self.batch_size = batch_size
		self.learn_step_cntr = 0
		self.time_step = 0
		self.warmup = warmup
		self.n_actions = n_actions
		self.update_actor_iter = update_actor_interval
		self.ep_greedy = 0.0

		self.actor = actor_nw(layer1_size, layer2_size, layer3_size, n_actions=n_actions, name='actor')
		self.critic_1 = critic_nw(layer1_size, layer2_size, layer3_size, name='critic_1')
		self.critic_2 = critic_nw(layer1_size, layer2_size, layer3_size, name='critic_2')

		self.target_actor = actor_nw(layer1_size, layer2_size, layer3_size, n_actions=n_actions, name='target_actor')
		self.target_critic_1 = critic_nw(layer1_size, layer2_size, layer3_size, name='target_critic_1')
		self.target_critic_2 = critic_nw(layer1_size, layer2_size, layer3_size, name='target_critic_2')

		self.actor.compile(optimizer=Adam(learning_rate=agent_lr), loss='mean')
		self.critic_1.compile(optimizer=Adam(learning_rate=critic_lr), loss='mean_squared_error')
		self.critic_2.compile(optimizer=Adam(learning_rate=critic_lr), loss='mean_squared_error')

		self.target_actor.compile(optimizer=Adam(learning_rate=agent_lr), loss='mean')
		self.target_critic_1.compile(optimizer=Adam(learning_rate=critic_lr), loss='mean_squared_error')
		self.target_critic_2.compile(optimizer=Adam(learning_rate=critic_lr), loss='mean_squared_error')

		self.noise = noise
		self.update_network_parameters(tau=1)

	def nw_action(self, observation):
		if self.time_step < self.warmup:
			mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
		else:
			state = tf.convert_to_tensor([observation], dtype=tf.float32)
			# returns a batch size of 1, want a scalar array
			mu = self.actor(state)[0]

# 		if np.random.rand() > self.ep_greedy:
# 			# Pick random action
# 			mu_ = (tf.random.uniform(shape=[2]) - tf.constant([0.5, 0.5]))*2
# 			# tf.convert_to_tensor(np.array([(np.random.rand()-0.5)*2, (np.random.rand()-0.5)*2]))

# 		else:
		mu_ = mu + np.random.normal(scale=self.noise)
		mu_ = tf.clip_by_value(mu_, self.min_action, self.max_action)
		

		self.time_step += 1


		return mu_

	def remember(self, state, action, reward, new_state, done):
		self.mem.store_in_buffer(state, action, reward, new_state, done)

	def learn(self):
		if self.mem.mem_cntr < self.batch_size:
			return

		states, actions, rewards, new_states, dones = self.mem.sample_buffer(self.batch_size)

		states = tf.convert_to_tensor(states, dtype=tf.float32)
		actions = tf.convert_to_tensor(actions, dtype=tf.float32)
		rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
		states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

		with tf.GradientTape(persistent=True) as tape:
			target_actions = self.target_actor(states_)
			target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

			target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

			q1 = self.critic_1(states, actions)
			q2 = self.critic_2(states, actions)
			q1_ = self.target_critic_1(states_, target_actions)
			q2_ = self.target_critic_2(states_, target_actions)
	 
			# Making 1 D for neural network
			q1 = tf.squeeze(q1, 1)
			q2 = tf.squeeze(q2, 1)

			q1_ = tf.squeeze(q1_, 1)
			q2_ = tf.squeeze(q2_, 1)


			critic_value_ = tf.math.minimum(q1_, q2_)

			target = rewards + self.gamma*critic_value_*(1-dones)
			critic_1_loss = keras.losses.MSE(target, q1)
			critic_2_loss = keras.losses.MSE(target, q2)

		critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
		critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

		self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
		self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

		self.learn_step_cntr += 1

		if self.learn_step_cntr % self.update_actor_iter != 0:
			return

		with tf.GradientTape() as tape:
			new_actions = self.actor(states)
			critic_1_value = self.critic_1(states, new_actions)
			actor_loss = -tf.math.reduce_mean(critic_1_value)

		actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

		self.update_network_parameters()

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		weights = []
		targets = self.target_actor.weights
		for i, weight in enumerate(self.actor.weights):
			weights.append(weight * tau + targets[i]*(1-tau))

		self.target_actor.set_weights(weights)

		weights = []
		targets = self.target_critic_1.weights
		for i, weight in enumerate(self.critic_1.weights):
			weights.append(weight * tau + targets[i]*(1-tau))

		self.target_critic_1.set_weights(weights)

		weights = []
		targets = self.target_critic_2.weights
		for i, weight in enumerate(self.critic_2.weights):
			weights.append(weight * tau + targets[i]*(1-tau))

		self.target_critic_2.set_weights(weights)

	def save_models(self):
		print('Saving models.......')
		self.actor.save_weights(self.actor.check_pt_file)
		self.target_actor.save_weights(self.target_actor.check_pt_file)
		self.critic_1.save_weights(self.critic_1.check_pt_file)
		self.critic_2.save_weights(self.critic_2.check_pt_file)
		self.target_critic_1.save_weights(self.target_critic_1.check_pt_file)
		self.target_critic_2.save_weights(self.target_critic_2.check_pt_file)

	def load_models(self):
		print('Loading models......')
		self.actor.load_weights(self.actor.check_pt_file)
		self.target_actor.load_weights(self.target_actor.check_pt_file)
		self.critic_1.load_weights(self.critic_1.check_pt_file)
		self.critic_2.load_weights(self.critic_2.check_pt_file)
		self.target_critic_1.load_weights(self.target_critic_1.check_pt_file)
		self.target_critic_2.load_weights(self.target_critic_2.check_pt_file)