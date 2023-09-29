from learners.UCBVI import *
from learners.UCBQL import *
from learners.UBEV import *

from utils_episodic import *

def run_exp(rendermode='', testName = "riverSwim", sup = ''):
	H=20
	nbEpisodes = 1* 10**3
	nbReplicates=1

	step = max((1, nbEpisodes // 1000))
	
	if testName == "riverSwim6":
		ns_river = 6
		env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05)
	elif testName == "riverSwim8":
		ns_river = 8
		env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05)
	elif testName == "4-room":
		env, nbS, nbA = buildGridworld(map_name=testName)
	else:
		print("Invalid test name, using riverSwim6 by default.")
		testName = "riverSwim6"
		ns_river = 6
		env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05)
	
	
	print("sup = ", sup)
	
	print("*********************************************")
	
	cumRewards = []
	names = []

	testName += "_H_" + str(H) + "_replicates_" + str(nbReplicates) + "_K_" + str(nbEpisodes)
	
	learner1 = UCB_QL( nbS,nbA, H, delta=0.05)
	names.append(learner1.name())
	cumRewards1 = cumulativeRewards(env,learner1,nbReplicates,H,nbEpisodes, step = step)
	cumRewards.append(cumRewards1)
	#pickle.dump(cumRewards1, open(("results/cumRewards_" + testName + "_" + learner1.name() + sup), 'wb'))
	
	#learner2 = UCBVI(nbS, nbA, H, delta=0.05)
	#names.append(learner2.name())
	#cumRewards2 = cumulativeRewards(env,learner2,nbReplicates,H,nbEpisodes, step = step)
	#cumRewards.append(cumRewards2)
	#pickle.dump(cumRewards2, open(("results/cumRewards_" + testName + "_" + learner2.name() + sup), 'wb'))
	
	learner3 = UBEV( nbS, nbA, H, delta=0.05)
	names.append(learner3.name())
	cumRewards3 = cumulativeRewards(env,learner3,nbReplicates,H,nbEpisodes, step = step)
	cumRewards.append(cumRewards3)
	#pickle.dump(cumRewards3, open(("results/cumRewards_" + testName + "_" + learner3.name() + sup), 'wb'))
	

	
#	inistate = env.reset()
#	optimal_value_sequence =  compute_Vstar_MultiRM(env, H, inistate=inistate) # To do: adapt to random initials states, fixed at 0 for now.
#	temp = [optimal_value_sequence[0]]
#	nbRM = len(optimal_value_sequence)
#	for k in range(1,nbEpisodes):
#		temp.append(temp[-1] + optimal_value_sequence[k%nbRM])
#	opti_reward = [temp]
	#print("Cumulative optimal Value: ", opti_reward[0][-1])

#	cumRewards.append(opti_reward)
	
	plotCumulativeRewards(names, cumRewards, nbEpisodes, H, testName, step = step)

	#plotCumulativeRegrets(names, cumRewards, nbEpisodes, H, testName)#, semilog=True, ysemilog=True)
	#plotCumulativeRegrets(names, cumRewards, nbEpisodes, H, testName, semilog=True)#, ysemilog=True)
	#plotCumulativeRegrets(names, cumRewards, nbEpisodes, H, testName, semilog=False, ysemilog=True)
	
	
	print("*********************************************")

run_exp(rendermode='', testName = "4-room",  sup = '_0')
run_exp(rendermode='', testName = "riverSwim6",  sup = '_0')
run_exp(rendermode='', testName = "riverSwim8",  sup = '_0')
