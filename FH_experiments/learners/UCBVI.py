import numpy as np
import copy as cp
import pylab as pls










class UCBVI():
	def __init__(self, nS, nA, H, delta = 0.05):
		self.nS = nS
		self.nA = nA
		self.delta = delta

		# Parameters:
		self.H = H
		self.N = np.zeros((nS, nA))
		self.Nsas =  np.zeros((nS, nA, nS))
		self.hatQ = None
		self.hatV = None
		self.R = np.zeros((self.nS, self.nA), dtype=float)
		self.t = 0

		# Set the initial state and last action:
		self.s = None
		self.last_action = -1

	
	def name(self):
		return "UCBVI"

	# return the current greedy policy associated with self.hatQ.
	def make_pol(self):
		res = np.argmax(self.hatQ)
		return res

	def bonus(self, s, a):
		L = np.log(5 * self.nS * self.nA * self.t / self.delta)
		return self.H * L * np.sqrt(1/self.N[s, a])#7 * self.H * L * np.sqrt(1/self.N[s, a])

	def reset(self, init):
		# Parameters:
		self.N = np.zeros((self.nS, self.nA))
		self.Nsas =  np.zeros((self.nS, self.nA, self.nS))
		self.R = np.zeros((self.nS, self.nA), dtype=float)
		self.hatQ = None
		self.hatV = None
		self.t = 0

		self.new_episode(init)
	
	def new_episode(self, init):
		# Set the initial state and last action:
		self.s = init
		self.last_action = -1

		hatP = np.zeros((self.nS, self.nA, self.nS), dtype=float)
		for s in range(self.nS):
			for a in range(self.nA):
				for ss in range(self.nS):
					hatP[s, a, ss] = self.Nsas[s, a, ss] / max((1, self.N[s, a]))
		

		# Update hatQ
		self.hatQ = np.zeros((self.H, self.nS, self.nA), dtype=float)
		self.hatV = np.zeros((self.H+1, self.nS), dtype=float)

		for h in range(self.H-1, 0, -1):
			for s in range(self.nS):
				for a in range(self.nA):
					if self.N[s, a] > 0:
						reward = self.R[s, a] / self.N[s, a]
						temp = reward + np.sum([hatP[s, a, ss] * self.hatV[h+1, ss] + self.bonus(s, a) for ss in range(self.nS)])
						self.hatQ[h, s, a] = np.min((self.hatQ[h, s, a], temp))
					else:
						self.hatQ[h, s, a] = self.H
					self.hatV[h, s] = np.max(self.hatQ[h, s])



	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state, h):
		self.t += 1

		# Naively select a random greedy action.
		action = np.argmax(self.hatQ[h, state])
		list_a = [action]
		for a in range(self.nA):
			if (self.hatQ[h, state, a] == self.hatQ[h, state, action]) and (a not in list_a):
				list_a.append(a)
		action = np.random.choice(list_a)

		return action
	
	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation, h):
		self.N[state, action] += 1
		self.Nsas[state, action, observation] += 1
		self.R[state, action] += reward







