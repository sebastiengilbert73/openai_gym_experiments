import gym
import numpy
import QLearning

class BlackjackQLearningEnvironment:
    def __init__(self):
        self.environment = gym.make('Blackjack-v0')
        self.environment.reset()

    def Reset(self):
        return self.ObservationAsInt( self.environment.reset() )

    def Step(self, action):
        observation, reward, done, info = self.environment.step(action)
        # Convert the tuple observation into an int
        return self.ObservationAsInt(observation), reward, done, info

    def ObservationAsInt(self, observationTuple):
        return observationTuple[0] * 22 + observationTuple[1] * 2 + observationTuple[2]

    def ObservationAsList(self, observationInt):
        observationList = [0, 0, 0]
        residual = observationInt
        observationList[2] = residual % 2
        residual -= observationList[2]
        observationList[1] = int( (residual % 22) / 2 )
        residual -= observationList[1] * 2
        observationList[0] = int( residual / 22 )
        return observationList


def main():
    print ("Blackjack-v0/useQLearning.py main()")
    blackjackEnv = BlackjackQLearningEnvironment()
    solver = QLearning.Solver(numberOfObservations=(32 * 11 * 2),
                              numberOfActions=2,
                              gamma=1.0,
                              learningRate=0.01,
                              environment=blackjackEnv,
                              defaultValue=0.01,
                              epsilonRampDownNumberOfEpisodes = 10000,
                              )
    solver.Solve(100000, writeToConsole=True)

    print (solver.Q[300])
    foundQ = solver.Q
    outputFile = open('foundQ.csv', 'w+')
    outputFile.write("observation, playerSum, dealerShowing, hasUsableAce, action, value\n")
    for obs in range(foundQ.shape[0]):
        obsList = blackjackEnv.ObservationAsList(obs)
        for action in range(foundQ.shape[1]):
            value = foundQ[obs, action]
            outputFile.write("{}, {}, {}, {}, {}, {}\n".format(obs, obsList[0], obsList[1], obsList[2], action, value))
    outputFile.close()

    outputFile2 = open('suttonBarto_p121.csv', 'w+')
    outputFile2.write("Dealer:,A,2,3,4,5,6,7,8,9,10\n")
    for thereIsAUsableAce in [1, 0]:
        for playerSum in range(21, 10, -1):
            outputFile2.write("{},".format(playerSum))
            for dealerShowing in range(1, 11):
                observation = blackjackEnv.ObservationAsInt((playerSum, dealerShowing, thereIsAUsableAce))
                bestAction = solver.BestAction(observation)
                outputFile2.write("{},".format(bestAction))
            outputFile2.write("\n")
    outputFile2.close()

    # Test it a large number of times
    rewardSum = 0
    numberOfEpisodes = 100000
    for episodeNdx in range(numberOfEpisodes):
        observation = blackjackEnv.Reset()
        done = False
        episodeRewardSum = 0
        while not done:
            bestAction = solver.BestAction(observation)
            observation, reward, done, info = blackjackEnv.Step(bestAction)
            episodeRewardSum += reward
        rewardSum += episodeRewardSum
    averageReward = rewardSum/numberOfEpisodes
    print ("averageReward = {}".format(averageReward))

if __name__ == '__main__':
    main()