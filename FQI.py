from sklearn.ensemble import RandomForestRegressor
import numpy as np

class QIterator:
    # Constructor
    def __init__(self, numActions, discount, numIter):
        self.Models=RandomForestRegressor()
        self.NumActions = numActions
        self.Discount = discount
        self.NumIter = numIter


    # Iterates the Q function based on observed action dependant transitions and rewards
    def iterateToHorizon(self, S, A, R, S_prime):
        numExamples = len(A)
        targets = np.array([0]*numExamples)
        A = np.array([A])
        for n in range(numExamples):
            targets[n] = R[n]
        self.iterateRegressor(S, A, targets)
        
        # Repeat iteration until Q function extends to numIter
        for h in range(self.NumIter):
            # Compute targets for next Q function
            for n in range(numExamples):
                Qprior = []
                for action in range(self.NumActions):
                   Qprior.append(self.predict(S_prime[n],action))
                targets[n] = R[n] + self.Discount*max(Qprior)
            self.iterateRegressor(S, A, targets)

    # Takes action dependant targets and features and fits a regressor
    def iterateRegressor(self, S,A, target):
        self.clearRegressor()
        features = np.concatenate([S, A.T], axis=1)
        self.Model.fit(features, target)

    # Returns the regressor predicted expected value for each action
    def predict(self, S,A):
        feature = np.append(S,A)
        return self.Model.predict([feature])

    # Replaces the regressor with a fresh random forest, used to iterate over Q functions
    def clearRegressor(self):
        self.Model = RandomForestRegressor()


class FittedQController:

    # Constructor
    def __init__(self, numActions, numFeatures, numIter, discountParameter):
        self.NumActions = numActions  
        self.NumFeatures = numFeatures  # The number of features in the input space
        self.NumIter = numIter  
        self.DiscountParameter = discountParameter  # The discount factor applied to future rewards
    
        self.QFunction = QIterator(numActions=numActions, discount=discountParameter, numIter = numIter)   

        self.S = np.zeros((0, numFeatures))
        self.A = np.zeros(0)
        self.R = np.zeros(0)
        self.S_prime = np.zeros((0, numFeatures))

    
    def addSamples(self, S,A,R,S_prime):
        self.A = np.append(self.A,A)
        self.R = np.append(self.R,R)
        self.S = np.concatenate((self.S, [S]), axis=0)
        self.S_prime = np.concatenate((self.S_prime, [S_prime]), axis=0)


        
    def queryForAction(self, state):
        qValues = []
        for action in range(self.NumActions):
            qValues.append(self.QFunction.predict(state,action))
        return qValues.index(max(qValues))


    # Runs the Q function through a round of FQI to generate a better greedy policy based on newly aquired experience
    def updatePolicyUsingExperience(self):
        self.QFunction.iterateToHorizon(S=self.S, A=self.A, R=self.R, S_prime=self.S_prime)


