import numpy as np
import copy as cp
import pylab as pls










class UBEV():
	def __init__(self, nS, nA, H, delta = 0.05):
		self.nS = nS
		self.nA = nA
		self.delta = delta

		# Parameters:
		self.H = H
		self.N = np.zeros((nS, nA))
		self.Nsas =  np.zeros((nS, nA, nS))
		self.R = np.zeros((self.nS, self.nA), dtype=float)
		self.hatQ = None
		self.t = 0

		# Set the initial state and last action:
		self.s = None
		self.last_action = -1

	def name(self):
		return "UBEV"

	# return the current greedy policy associated with self.hatQ.
	def make_pol(self):
		res = np.argmax(self.hatQ)
		return res

	def bonus(self, s, a):
		L = np.log(5 * self.current_U * self.nS * self.nA * self.t / self.delta)
		return self.H * L * np.sqrt(1/self.N[s, a])#7 * self.H * L * np.sqrt(1/self.N[s, a])

	def reset(self, init):
		# Parameters:
		self.N = np.zeros((self.nS, self.nA))
		self.Nsas =  np.zeros((self.nS, self.nA, self.nS))
		self.R = np.zeros((self.nS, self.nA), dtype=float)
		self.hatQ = None
		self.t = 0
		self.current_n = None

		self.new_episode(init)
	
	def new_episode(self, init):
		# Set the initial state and last action:
		self.s = init
		self.last_action = -1
		

		# Update hatQ
		self.hatQ = np.zeros((self.H, self.nS, self.nA), dtype=float)
		self.hatV = np.zeros((self.H+1, self.nS), dtype=float)

		phi = np.zeros((self.nS, self.nA), dtype = float)
		for s in range(self.nS):
			for a in range(self.nA):
				temp = 2 * np.log(np.log(max((np.exp(1), self.N[s, a])))) + np.log(18 * self.nS * self.nA * self.H / self.delta)
				phi[s, a] = np.sqrt(temp / max((1, self.N[s, a])))

		for h in range(self.H-1, 0, -1):
			for s in range(self.nS):
				for a in range(self.nA):
					reward = self.R[s, a] / max((1, self.N[s, a]))

					hatVnext = sum([self.Nsas[s, a, ss] * self.hatV[h+1, ss] / max((1, self.N[s, a])) for ss in range(self.nS)])

					temp = hatVnext + (self.H - h) * phi[s, a]
					temp2 = np.max((self.hatV[h+1]))
					self.hatQ[h, s, a] = reward + np.min((temp, temp2))
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

