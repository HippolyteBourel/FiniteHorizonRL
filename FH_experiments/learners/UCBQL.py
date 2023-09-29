import numpy as np
import copy as cp
import pylab as pls










class UCB_QL():
	def __init__(self, nS, nA, H, c = 0.1, delta = 0.05):
		self.nS = nS
		self.nA = nA
		self.c = c
		self.delta = delta

		# Parameters:
		self.H = H
		self.N = np.zeros((nS, nA))
		self.hatQ = np.full((self.H, self.nS, self.nA), self.H, dtype=float)
		self.hatV = np.full((self.H+1 , self.nS), 0, dtype=float)
		self.t = 0

		# Set the initial state and last action:
		self.s = None
		self.last_action = -1
	
	def name(self):
		return "UCB-QL"

	# return the current greedy policy associated with self.hatQ.
	def make_pol(self):
		res = np.argmax(self.hatQ)
		return res
	
	def reset(self, init):
		# Parameters:
		self.N = np.zeros((self.nS, self.nA))
		self.hatQ = np.full((self.H, self.nS, self.nA), self.H, dtype=float)
		self.hatV = np.zeros((self.H+1, self.nS), dtype=float)
		self.t = 0
	
	def new_episode(self, init):
		# Parameters:

		# Set the initial state and last action:
		self.s = init
		self.last_action = -1

	# To chose an action for a given state.
	def play(self,state, h):
		
		# Naively select a random greedy action.
		action = np.argmax(self.hatQ[h, state])
		list_a = [action]
		for a in range(self.nA):
			if (self.hatQ[h, state, a] == self.hatQ[h, state, action]) and (a not in list_a):
				list_a.append(a)
		action = np.random.choice(list_a)

		self.N[state, action] += 1
		self.s = state
		self.last_action = action

		return action
	
	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation, h):
		self.t += 1
		
		# Computing the constant used for the Q-learning.
		if h >= 1:
			N = max((1, self.N[state, action]))
			alpha_t = (self.H + 1) / (self.H + N)
			iota = 1#np.log(N)#np.log(self.t)
			b_t = self.c * np.sqrt(self.H**3 * iota / N)

			# Udpate Q-learning for all states of all RMs.
			self.hatQ[h, state, action] = (1 - alpha_t) * self.hatQ[h, state, action] + alpha_t * (reward + b_t + self.hatV[h+1, observation])
			self.hatV[h, state] = np.min((self.H, np.max(self.hatQ[h, state])))


