import numpy as np
from FQI import FittedQController, QIterator

actionMap = {'eat':0,'sleep':1,'play':2}
#training file name here
fname = 'example.txt' 
controller = FittedQController(numActions = 3, numFeatures=4, numIter = 3, discountParameter=0.3)

# parse input data
with open(fname) as f:
    content = f.readlines()
    for line in content:
       sars = line.split(' ')
       a = sars[1]
       action = actionMap[a]
       reward = float(sars[2])
       state = map(lambda x: float(x),sars[0].split(','))
       sprime = map(lambda x: float(x),sars[3].split(','))   
       controller.addSamples(state,action,reward,sprime)

#train
controller.updatePolicyUsingExperience()

# query for new state, see what is the best action for this state, in this example
# it is state (3,2,3,4)
action = controller.queryForAction([1,2,3,4])
for a,num in actionMap.iteritems():
    if num == action:
        print a
