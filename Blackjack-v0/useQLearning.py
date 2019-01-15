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


def main():
    print ("Blackjack-v0/useQLearning.py main()")
    blackjackEnv = BlackjackQLearningEnvironment()
    solver = QLearning.Solver(numberOfObservations=(32 * 11 * 2),
                              numberOfActions=2,
                              gamma=0.99,
                              learningRate=0.3,
                              environment=blackjackEnv
                              )
    solver.Solve(100000, writeToConsole=True)

    # Test it a large number of times
    rewardSum = 0
    numberOfEpisodes = 10000
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