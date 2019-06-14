import os, sys, time  
import numpy as np
import tensorflow as tf
import random
from keras.utils import np_utils,to_categorical
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import copy
import matplotlib.pyplot as plt

n_agent=10
n_resource=3
resource=[]
for i in range(n_resource):
	resource.append(np.random.rand(2))
ant=[]
size=[]
speed=[]
for i in range(n_agent):
	ant.append(np.random.rand(2))
	size.append(0.01+np.random.rand()*0.04)
	speed.append(0.01+size[i])

def get_obs(ant,resource,si,sp,n_agent):
	state=[]
	for i in range(n_agent):
		h=[]
		h.append(ant[i][0])
		h.append(ant[i][1])
		h.append(si[i])
		h.append(sp[i])
		j=0
		mi = 10
		for k in range(len(resource)):
			t = (resource[k][0]-ant[i][0])**2+(resource[k][1]-ant[i][1])**2
			if t<mi:
				j = k
				mi = t
		h.append(resource[j][0])
		h.append(resource[j][1])
		state.append(h)
	return state

def step(ant,resource,n_resource,n_agent,size,speed,action):
	re=[0]*n_agent
	for i in range(n_agent):
		if action[i]==1:
			ant[i][0]-=speed[i]
			if ant[i][0]<0:
				ant[i][0]=0
		if action[i]==2:
			ant[i][0]+=speed[i]
			if ant[i][0]>1:
				ant[i][0]=1
		if action[i]==3:
			ant[i][1]-=speed[i]
			if ant[i][1]<0:
				ant[i][1]=0
		if action[i]==4:
			ant[i][1]+=speed[i]
			if ant[i][1]>1:
				ant[i][1]=1
	for i in range(n_resource):
		for j in range(n_agent):
			if (resource[i][0]-ant[j][0])**2+(resource[i][1]-ant[j][1])**2<size[j]**2:
				re[j]=1
				resource[i]=np.random.rand(2)
				size[j]=min(size[j]+0.005,0.15)
				speed[j]=0.01+size[j]
				break

	return ant,resource,size,speed,re

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

			self.saver = tf.train.Saver(self.W,max_to_keep=3000)
			
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
T = 50
totalTime = 0
GAMMA = 0.98
n_episode = 100000
max_steps = 1000
i_episode = 0
n_actions = 5
n_signal = 4
render = False

meta_Pi = PPOPolicyNetwork(num_features=8, num_actions=n_signal,layer_size=128,epsilon=0.2,learning_rate=0.0003)
meta_V = ValueNetwork(num_features=8, hidden_size=128, learning_rate=0.001)

Pi = []
V = []
for i in range(n_signal):
	Pi.append(PPOPolicyNetwork(num_features=6, num_actions=n_actions,layer_size=256,epsilon=0.2,learning_rate=0.0003))
	V.append(ValueNetwork(num_features=6, hidden_size=256, learning_rate=0.001))

while i_episode<n_episode:
	i_episode+=1

	avg = [0]*n_agent
	u_bar = [0]*n_agent
	utili = [0]*n_agent
	u = [[] for _ in range(n_agent)]
	max_u = 0.15

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
	resource=[]
	for i in range(n_resource):
		resource.append(np.random.rand(2))
	ant=[]
	size=[]
	speed=[]
	su=[0]*n_agent
	for i in range(n_agent):
		ant.append(np.random.rand(2))
		size.append(0.01+np.random.rand()*0.04)
		speed.append(0.01+size[i])
	su = np.array(su)

	obs = get_obs(ant,resource,size,speed,n_agent)

	while steps<max_steps:

		if steps%T==0:
			for i in range(n_agent):
				h = copy.deepcopy(obs[i])
				h.append(rat[i])
				h.append(utili[i])
				p_z = meta_Pi.get_dist(np.array([h]))[0]
				z = np.random.choice(range(n_signal), p=p_z)
				signal[i]=z
				meta_z[i].append(to_categorical(z,n_signal))
				meta_states[i].append(h)

		steps+=1
		action=[]
		for i in range(n_agent):
			h = copy.deepcopy(obs[i])
			p = Pi[signal[i]].get_dist(np.array([h]))[0]
			action.append(np.random.choice(range(n_actions), p=p))
			ep_states[i].append(h)
			ep_actions[i].append(to_categorical(action[i],n_actions))
		
		ant,resource,size,speed,rewards=step(ant,resource,n_resource,n_agent,size,speed,action)
		
		su+=np.array(rewards)
		score += sum(rewards)
		obs = get_obs(ant,resource,size,speed,n_agent)

		for i in range(n_agent):
			u[i].append(rewards[i])
			u_bar[i] = sum(u[i])/len(u[i])
		'''
		avg=copy.deepcopy(u_bar)
		for j in range(10):
			for i in range(n_agent):
				s=0
				for k in range(3):
					m=np.random.randint(0,n_agent)
					s+=avg[m]
				avg[i]=(avg[i]*0.02+(avg[i]+s)/(3+1)*0.98)+(np.random.rand()-0.5)*0.0001
		'''
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
				p_z = meta_Pi.get_dist(np.array([h]))[0]
				r_p = p_z[signal[i]]
				ep_rewards[i].append(r_p)

		if steps%T==0:
			for i in range(n_agent):
				meta_rewards[i].append(utili[i]/(0.1+abs(rat[i])))
				ep_actions[i] = np.array(ep_actions[i])
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])
				targets = discount_rewards(ep_rewards[i],GAMMA)
				V[signal[i]].update(ep_states[i], targets)
				vs = V[signal[i]].get(ep_states[i])
				ep_advantages = targets - vs
				ep_advantages = (ep_advantages - np.mean(ep_advantages))/(np.std(ep_advantages)+0.0000000001)
				Pi[signal[i]].update(ep_states[i], ep_actions[i], ep_advantages)
				
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
		meta_V.update(meta_states[i], meta_rewards[i])
		meta_advantages = meta_rewards[i]-meta_V.get(meta_states[i])
		meta_Pi.update(meta_states[i], meta_z[i], meta_advantages)
	print(i_episode)
	print(score/max_steps)
	print(su)
	uti = np.array(su)/max_steps