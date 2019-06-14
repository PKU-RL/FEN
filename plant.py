import os, sys, time  
import numpy as np
import tensorflow as tf
import random
from keras.utils import np_utils,to_categorical
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import copy
import matplotlib.pyplot as plt
n_agent=5
env = np.zeros((12,12))
n_resource=8
resource=[]
resource_type=[]
ant = []
possession=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
requirement=[[2, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 2]]
global number
number = 0

for i in range(n_agent):
	ant.append(np.random.randint(2,10,2))
	env[ant[i][0]][ant[i][1]]=1

for i in range(n_resource):
	resource.append(np.random.randint(3,9,2))
	resource_type.append(np.random.randint(3))

def get_obs(ant,resource,resource_type,env,possession,requirement):
	n_agent = 5
	n_resource = 8
	h=[]
	re_map=np.zeros((12,12))
	for i in range(n_resource):
		re_map[resource[i][0]][resource[i][1]]=resource_type[i]+1
	for k in range(n_agent):
		state = []
		state.append(ant[k][0])
		state.append(ant[k][1])
		for i in range(3):
			state.append(possession[k][i])
		for i in range(-2,3):
			for j in range(-2,3):
				state.append(re_map[ant[k][0]+i][ant[k][1]+j])
		h.append(state)
	return h

def step(env,ant,action,resource,resource_type,possession,requirement):
	n_agent = 5
	n_resource = 8
	next_ant=[]
	global number
	for i in range(n_agent):
		x=ant[i][0]
		y=ant[i][1]
		if action[i]==0:
			next_ant.append([x,y])
		if action[i]==1:
			x=x-1
			if x==1:
				next_ant.append([x+1,y])
				continue
			if env[x][y]!=1:
				env[x][y]=1
				next_ant.append([x,y])
			else:
				next_ant.append([x+1,y])
		if action[i]==2:
			x=x+1
			if x==10:
				next_ant.append([x-1,y])
				continue
			if env[x][y]!=1:
				env[x][y]=1
				next_ant.append([x,y])
			else:
				next_ant.append([x-1,y])
		if action[i]==3:
			y=y-1
			if y==1:
				next_ant.append([x,y+1])
				continue
			if env[x][y]!=1:
				env[x][y]=1
				next_ant.append([x,y])
			else:
				next_ant.append([x,y+1])
		if action[i]==4:
			y=y+1
			if y==10:
				next_ant.append([x,y-1])
				continue
			if env[x][y]!=1:
				env[x][y]=1
				next_ant.append([x,y])
			else:
				next_ant.append([x,y-1])
	ant = next_ant
	env*=0
	re = [0]*n_agent
	for i in range(n_agent):
		env[ant[i][0]][ant[i][1]]=1

	for j in range(n_resource):
		for i in range(n_agent):
			if (ant[i][0]==resource[j][0])&(ant[i][1]==resource[j][1]):
				resource[j]=np.random.randint(3,9,2)
				possession[i][resource_type[j]]+=1
				if possession[i][resource_type[j]]<requirement[i][resource_type[j]]:
					re[i]+=0.1
				resource_type[j] = np.random.randint(3)
				number+=1
				break
	for i in range(n_agent):
		x=1000
		for j in range(3):
			if requirement[i][j]==0:
				continue
			else:
				t = int(possession[i][j]/requirement[i][j])
				if t<x:
					x=t
		re[i]+=(x)
		for j in range(3):
			possession[i][j]-=requirement[i][j]*x

	return env,ant,resource,resource_type,possession,re
class ValueNetwork():
	def __init__(self, num_features, hidden_size, learning_rate=.01):
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.tf_graph = tf.Graph()
		with self.tf_graph.as_default():
			self.session = tf.Session()

			self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
			self.W = [
				tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
				tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
				tf.get_variable("W3", shape=[self.hidden_size, 1])
			]
			self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
			self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]))
			self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]), [-1])

			self.rollout = tf.placeholder(shape=[None], dtype=tf.float32)
			self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
			self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.minimize = self.grad_optimizer.minimize(self.loss)

			init = tf.global_variables_initializer()
			self.session.run(init)

	def get(self, states):
		value = self.session.run(self.output, feed_dict={self.observations: states})
		return value

	def update(self, states, discounted_rewards):
		_, loss = self.session.run([self.minimize, self.loss], feed_dict={
			self.observations: states, self.rollout: discounted_rewards
		})


class PPOPolicyNetwork():
	def __init__(self, num_features, layer_size, num_actions, epsilon=.2,
				 learning_rate=9e-4):
		self.tf_graph = tf.Graph()

		with self.tf_graph.as_default():
			self.session = tf.Session()

			self.observations = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
			self.W = [
				tf.get_variable("W1", shape=[num_features, layer_size]),
				tf.get_variable("W2", shape=[layer_size, layer_size]),
				tf.get_variable("W3", shape=[layer_size, num_actions])
			]

			self.saver = tf.train.Saver(self.W,max_to_keep=1000)
			
			self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
			self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]))
			self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]))

			self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

			self.chosen_actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
			self.old_probabilities = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

			self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions*self.output, axis=1)
			self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions*self.old_probabilities, axis=1)

			self.ratio = self.new_responsible_outputs/self.old_responsible_outputs

			self.loss = tf.reshape(
							tf.minimum(
								tf.multiply(self.ratio, self.advantages), 
								tf.multiply(tf.clip_by_value(self.ratio, 1-epsilon, 1+epsilon), self.advantages)),
							[-1]
						) - 0.03*self.new_responsible_outputs*tf.log(self.new_responsible_outputs + 1e-10)
			self.loss = -tf.reduce_mean(self.loss)

			self.W0_grad = tf.placeholder(dtype=tf.float32)
			self.W1_grad = tf.placeholder(dtype=tf.float32)
			self.W2_grad = tf.placeholder(dtype=tf.float32)

			self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad]
			self.trainable_vars = self.W
			self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

			self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
			self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
			init = tf.global_variables_initializer()
			self.session.run(init)

	def get_dist(self, states):
		dist = self.session.run(self.output, feed_dict={self.observations: states})
		return dist

	def update(self, states, chosen_actions, ep_advantages):
		old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
		self.session.run(self.apply_grad, feed_dict={
			self.W0_grad: self.gradients[0][0],
			self.W1_grad: self.gradients[1][0],
			self.W2_grad: self.gradients[2][0],

		})
		self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
			self.observations: states,
			self.advantages: ep_advantages,
			self.chosen_actions: chosen_actions,
			self.old_probabilities: old_probabilities
		})
	def save_w(self,name):
		self.saver.save(self.session,name+'.ckpt')
	def restore_w(self,name):
		self.saver.restore(self.session,name+'.ckpt')

def discount_rewards(rewards,gamma):
		running_total = 0
		discounted = np.zeros_like(rewards)
		for r in reversed(range(len(rewards))):
			running_total = running_total *gamma + rewards[r]
			discounted[r] = running_total
		return discounted

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)
T = 500
totalTime = 0
GAMMA = 0.98
n_episode = 100000
max_steps = 10000
i_episode = 0
n_actions = 5
n_signal = 4
render = False

meta_Pi=[]
meta_V=[]
for i in range(n_agent):
	meta_Pi.append(PPOPolicyNetwork(num_features=32, num_actions=n_signal,layer_size=256,epsilon=0.2,learning_rate=0.0003))
	meta_V.append(ValueNetwork(num_features=32, hidden_size=256, learning_rate=0.001))

Pi=[[] for _ in range(n_agent)]
V=[[] for _ in range(n_agent)]
for i in range(n_agent):
	for j in range(n_signal):
		Pi[i].append(PPOPolicyNetwork(num_features=30, num_actions=n_actions,layer_size=256,epsilon=0.2,learning_rate=0.0003))
		V[i].append(ValueNetwork(num_features=30, hidden_size=256, learning_rate=0.001))

while i_episode<n_episode:
	i_episode+=1
	number = 0
	avg = [0]*n_agent
	u_bar = [0]*n_agent
	utili = [0]*n_agent
	u = [[] for _ in range(n_agent)]
	max_u = 0.003

	ep_actions  = [[] for _ in range(n_agent)]
	ep_rewards  = [[] for _ in range(n_agent)]
	ep_states   = [[] for _ in range(n_agent)]

	meta_z  = [[] for _ in range(n_agent)]
	meta_rewards  = [[] for _ in range(n_agent)]
	meta_states  = [[] for _ in range(n_agent)]

	signal = [0]*n_agent
	rat = [0.0]*n_agent

	score=0
	steps=0
	env = np.zeros((12,12))
	resource=[]
	resource_type=[]
	ant = []
	possession=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
	for i in range(n_agent):
		ant.append(np.random.randint(2,10,2))
		env[ant[i][0]][ant[i][1]]=1

	for i in range(n_resource):
		resource.append(np.random.randint(3,9,2))
		resource_type.append(np.random.randint(3))
	su=[0]*n_agent
	ac=[0]*n_actions
	su = np.array(su)
	obs = get_obs(ant,resource,resource_type,env,possession,requirement)
	while steps<max_steps:
		if number>700:
			break
		if steps%T==0:
			for i in range(n_agent):
				h = copy.deepcopy(obs[i])
				h.append(rat[i])
				h.append(utili[i])
				p_z = meta_Pi[i].get_dist(np.array([h]))[0]
				z = np.random.choice(range(n_signal), p=p_z)
				signal[i]=z
				meta_z[i].append(to_categorical(z,n_signal))
				meta_states[i].append(h)

		steps+=1
		action=[]
		for i in range(n_agent):
			h = copy.deepcopy(obs[i])
			p = Pi[i][signal[i]].get_dist(np.array([h]))[0]
			action.append(np.random.choice(range(n_actions), p=p))
			ep_states[i].append(h)
			ep_actions[i].append(to_categorical(action[i],n_actions))
		
		env,ant,resource,resource_type,possession,rewards=step(env,ant,action,resource,resource_type,possession,requirement)
		su+=np.array(rewards, dtype=np.int)
		score += sum(rewards)
		obs = get_obs(ant,resource,resource_type,env,possession,requirement)
		for i in range(n_agent):
			u[i].append(rewards[i])
			u_bar[i] = sum(u[i])/len(u[i])
		for i in range(n_agent):
			avg[i] = sum(u_bar)/len(u_bar)
			if avg[i]!=0:
				rat[i]=(u_bar[i]-avg[i])/avg[i]
			else:
				rat[i]=0
			utili[i] = min(1,avg[i]/max_u)

		for i in range(n_agent):
			if signal[i]==0:
				ep_rewards[i].append(rewards[i])
			else:
				h=copy.deepcopy(obs[i])
				h.append(rat[i])
				h.append(utili[i])
				p_z = meta_Pi[i].get_dist(np.array([h]))[0]
				r_p = p_z[signal[i]]
				ep_rewards[i].append(r_p)

		if steps%T==0:
			for i in range(n_agent):
				meta_rewards[i].append(utili[i]/(0.1+abs(rat[i])))
				if signal[i]==0:
					continue
				ep_actions[i] = np.array(ep_actions[i])
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])
				targets = discount_rewards(ep_rewards[i],GAMMA)
				V[i][signal[i]].update(ep_states[i], targets)
				vs = V[i][signal[i]].get(ep_states[i])
				ep_advantages = targets - vs
				ep_advantages = (ep_advantages - np.mean(ep_advantages))/(np.std(ep_advantages)+0.0000000001)
				Pi[i][signal[i]].update(ep_states[i], ep_actions[i], ep_advantages)
				
			ep_actions  = [[] for _ in range(n_agent)]
			ep_rewards  = [[] for _ in range(n_agent)]
			ep_states  = [[] for _ in range(n_agent)]
		
		if render:
			for i in range(n_agent):
				theta = np.arange(0, 2*np.pi, 0.01)
				x = ant[i][0] + size[i] * np.cos(theta)
				y = ant[i][1] + size[i] * np.sin(theta)
				plt.plot(x, y)
			for i in range(n_resource):
				plt.scatter(resource[i][0], resource[i][1], color = 'green')
			plt.axis("off")
			plt.axis("equal")
			plt.xlim(0 , 1)
			plt.ylim(0 , 1)
			plt.ion()
			plt.pause(0.1)
			plt.close()

	for i in range(n_agent):
		if len(meta_rewards[i])==0:
			continue
		meta_z[i] = np.array(meta_z[i])
		meta_rewards[i] = np.array(meta_rewards[i])
		meta_states[i] = np.array(meta_states[i])
		meta_V[i].update(meta_states[i], meta_rewards[i])
		meta_advantages = meta_rewards[i]-meta_V[i].get(meta_states[i])
		meta_Pi[i].update(meta_states[i], meta_z[i], meta_advantages)

	print(i_episode)
	print(score/max_steps)
	print(su)
	uti = np.array(su)/max_steps