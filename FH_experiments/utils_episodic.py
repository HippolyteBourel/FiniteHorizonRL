from environments.discreteMDP import *
from environments.gridworld import *
import pylab as pl
import gym
from gym.envs.registration import  register
import numpy as np

def buildRiverSwim(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
	register(
		id='RiverSwim-v0',
		entry_point='environments.discreteMDP:RiverSwim',
		max_episode_steps=max_steps,
		reward_threshold=reward_threshold,
		kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
				'rewardL': rewardL, 'rewardR':rewardR, }
	)

	return gym.make('RiverSwim-v0'), nbStates,2



def buildGridworld(sizeX=7,sizeY=7,map_name="4-room",rewardStd=0., initialSingleStateDistribution=False,max_steps=np.infty,reward_threshold=np.infty):
	register(
		id='Gridworld'+map_name+'-v0',
		entry_point='environments.gridworld:GridWorld',
		max_episode_steps=max_steps,
		reward_threshold=reward_threshold,
		kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':map_name,'rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution}
	)
	g = gym.make('Gridworld'+map_name+'-v0')
	return g, g.env.nS, 4



def cumulativeRewards(env,learner,nbReplicates,episodeHorizon,nbEpisode,step=1):
	cumRewards = []
	for i_episode in range(nbReplicates):
		observation = env.reset()
		learner.reset(observation)
		cumreward = 0.
		cumrewards = []
		print("New initialization of ", learner.name())
		print("Initial state:" + str(observation))
		print("Nb of episode: ", nbEpisode, " with episode Horizon = ", episodeHorizon)
		for episode in range(nbEpisode):
			for h in range(episodeHorizon):
				state = observation
				action = learner.play(state, h)  # Get action
				observation, reward, done, info = env.step(action)
				learner.update(state, action, reward, observation, h)  # Update learners
				cumreward += reward
				if done or h >= episodeHorizon: # Useless
					print("Episode finished after {} timesteps".format(h + 1))
					break
			if episode%step == 0:
				cumrewards.append(cumreward)
			observation = env.reset()
			learner.new_episode(init = observation)
		cumRewards.append(cumrewards)
		print("Cumreward:" + str(cumreward))
	return cumRewards


# To plot the regret over timepsteps of n different learner,names is the list of their names, cumulativerewards_ is the list of their cumulative
# reward + the one of the optimal policy used to compute the regret. Some optimal policy are given in the Optimal file in the learner folder.
def plotCumulativeRegrets(names,cumulativerewards_, nbEpisode, episodeHorizon, testName = "riverSwim", semilog = False, ysemilog = False):
	#print(len(cumulativerewards_[0]), len(cumulativerewards_))#[0])#print(len(cumulativerewards_[0]), cumulativerewards_[0])
	nbFigure = pl.gcf().number+1
	pl.figure(nbFigure)
	textfile = "results/Regret-"
	colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
	avgcum_rs= [ np.mean(cumulativerewards_[-1], axis=0) - np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_) - 1)]
	std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_) - 1)]
	for i in range(len(cumulativerewards_) - 1):
		pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
		step=nbEpisode//10
		pl.errorbar(np.arange(0,nbEpisode,step), avgcum_rs[i][0:nbEpisode:step], std_cum_rs[i][0:nbEpisode:step], color=colors[i%len(colors)], linestyle='None', capsize=10)
		textfile+=names[i]+"-"
		print(names[i], ' has regret ', avgcum_rs[i][-1], ' after ', len(avgcum_rs[i]), ' episodes with variance ', std_cum_rs[i][-1])
		#pl.show()
	pl.legend()
	pl.xlabel("Number of episodes", fontsize=13, fontname = "Arial")
	pl.ylabel("Regret", fontsize=13, fontname = "Arial")
	pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
	if semilog:
		pl.xscale('log')
		textfile += "_xsemilog"
	else:
		pl.xlim(0, nbEpisode)
	if ysemilog:
		pl.yscale('log')
		textfile += "_ysemilog"
		pl.ylim(1)
	else:
		pl.ylim(0)
	#pl.savefig(textfile + testName + '.png')
	pl.savefig(textfile + testName + '.pdf')




def plotCumulativeRewards(names,cumulativerewards_, nbEpisode, episodeHorizon, testName = "riverSwim", semilog = False, ysemilog = False, step = 1):
	#print(len(cumulativerewards_[0]), len(cumulativerewards_))#[0])#print(len(cumulativerewards_[0]), cumulativerewards_[0])
	nbFigure = pl.gcf().number+1
	nbEpisode = nbEpisode//step
	pl.figure(nbFigure)
	textfile = "results/Reward-"
	colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
	avgcum_rs= [ np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
	std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_))]
	for i in range(len(cumulativerewards_)):
		pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
		step_error=nbEpisode//10
		pl.errorbar(np.arange(0,nbEpisode,step_error), avgcum_rs[i][0:nbEpisode:step_error], std_cum_rs[i][0:nbEpisode:step_error], color=colors[i%len(colors)], linestyle='None', capsize=10)
		textfile+=names[i]+"-"
		print(names[i], ' has reward ', avgcum_rs[i][-1], ' after ', len(avgcum_rs[i])*step, ' episodes with variance ', std_cum_rs[i][-1])
		#pl.show()
	pl.legend()
	temp = "Number of episodes //" + str(step)
	pl.xlabel(temp, fontsize=13, fontname = "Arial")
	pl.ylabel("Rewards", fontsize=13, fontname = "Arial")
	pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
	if semilog:
		pl.xscale('log')
		textfile += "_xsemilog"
	else:
		pl.xlim(0, nbEpisode)
	if ysemilog:
		pl.yscale('log')
		textfile += "_ysemilog"
		pl.ylim(1)
	else:
		pl.ylim(0)
	#pl.savefig(textfile + testName + '.png')
	pl.savefig(textfile + testName + '.pdf')






def policy2int(pol, base = 2):#################################### Complete -> see Orel
	return 2


def int2policy(pol, base = 2):#################################### Complete -> see Orel
	return []




# Norme 1 of the difference between 2 vectors of same size.
def diffNorme1(v1, v2):
	res = 0
	for i in range(len(v1)):
		res += abs(v1[i] - v2[i])
	return res

	

# Doesn't work for the gridworlds -> need to use the mapping..
def make_transitions(P, nS, nA):
	res = np.zeros((nS, nA, nS))
	for s in range(nS):
		for a in range(nA):
			for e in P[s][a]:
				res[s, a, e[1]] = e[0]
	return res












# Value iteration
def value_iter(env, episodeHorizon, mean_reward = np.array([None])):
	if mean_reward.any() == None:
		mean_reward = env.mean_reward
	niter = 0
	policy = np.zeros((episodeHorizon, env.nS), dtype=int)
	u0 = np.zeros(env.nS)
	u1 = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	for h in episodeHorizon:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = mean_reward[s, a] + sum([u * p for (u, p) in zip(u0, P[s, a])])
				if (a == 0) or (temp > u1[s]):
					u1[s] = temp
					policy[episodeHorizon - h - 1, s] = a
		u0 = u1
		u1 = np.zeros(env.nS)
	u0, policy





def make_mean_reward(env):
	res = np.zeros((env.nS, env.nA))
	for s in range(env.nS):
		for a in range(env.nA):
			res[s, a] = env.getReward(s, a)
	return res

def compute_gstar(env, H, inistate = 0):
	mean_reward = make_mean_reward(env)
	#policy = np.zeros((H, env.nS), dtype=int)
	u0 = np.zeros(env.nS)
	u1 = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	for h in range(H):
		for s in range(env.nS):
			for a in range(env.nA):
				temp = mean_reward[s, a] + sum([u * p for (u, p) in zip(u0, P[s, a])])
				if (a == 0) or (temp > u1[s]):
					u1[s] = temp
					#policy[H - h - 1, s] = a
		u0 = u1
		u1 = np.zeros(env.nS)
	return max(u0) - u0[inistate]


def make_cp_transitions(RM, P, nS, nA):
	nQ = RM.nb_states
	res = np.zeros((nQ, nS, nA, nQ, nS),dtype=float)
	for q in range(nQ):
		for s in range(nS):
			for a in range(nA):
				for qq in range(nQ):
					for ss in range(nS):
						event = RM.events[s, a]
						if event == None:
							q_transition = q
						else:
							q_transition = RM.transitions[q, event]
						if qq == q_transition:
							res[q, s, a, qq, ss] = P[s, a, ss]
	return res






def compute_Vstar(env, H, inistate = 0):
	mean_reward = make_mean_reward(env)
	#policy = np.zeros((H, env.nS), dtype=int)
	u0 = np.zeros(env.nS)
	u1 = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	for h in range(H):
		for s in range(env.nS):
			for a in range(env.nA):
				temp = mean_reward[s, a] + sum([u * p for (u, p) in zip(u0, P[s, a])])
				if (a == 0) or (temp > u1[s]):
					u1[s] = temp
					#policy[H - h - 1, s] = a
		u0 = u1
		u1 = np.zeros(env.nS)
	return u0[inistate]

