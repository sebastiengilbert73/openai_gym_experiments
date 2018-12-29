import argparse
import gym
import torch
from collections import OrderedDict
import copy
import os
import math
import numpy
import sys

"""
>>> acrobotEnv = gym.make('Acrobot-v1')
WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype.
>>> acrobotEnv.action_space
Discrete(3)
>>> acrobotEnv.observation_space
Box(6,)
>>> acrobotEnv.observation_space.low
array([ -1.      ,  -1.      ,  -1.      ,  -1.      , -12.566371,
       -28.274334], dtype=float32)
>>> acrobotEnv.observation_space.high
array([ 1.      ,  1.      ,  1.      ,  1.      , 12.566371, 28.274334],
      dtype=float32)
"""


class NeuralNet(torch.nn.Module):
    def __init__(self, hiddenLayerWidths=(8,5)):
        super(NeuralNet, self).__init__()
        layersDict = OrderedDict()
        for hiddenLayerNdx in range(len(hiddenLayerWidths) + 1):
            if hiddenLayerNdx == 0:
                numberOfInputs = 6 # acrobotEnv.observation_space
            else:
                numberOfInputs = hiddenLayerWidths[hiddenLayerNdx - 1]

            if hiddenLayerNdx == len(hiddenLayerWidths):
                numberOfOutputs = 3 # acrobotEnv.action_space, discrete
            else:
                numberOfOutputs = hiddenLayerWidths[hiddenLayerNdx]
            layersDict['layer' + str(hiddenLayerNdx)] = self.FullyConnectedLayer(numberOfInputs, numberOfOutputs)

        self.layers = torch.nn.Sequential(layersDict)
        self.apply(init_weights)
        self.observation_low = [ -1., -1., -1., -1., -12.566371, -28.274334]
        self.observation_high = [ 1., 1., 1., 1., 12.566371, 28.274334]


    def forward(self, inputs):
        dataState = inputs
        for layerNdx in range(len(self.layers)):
            dataState = self.layers[layerNdx](dataState)
        return torch.nn.functional.softmax(dataState, dim=0)

    def FullyConnectedLayer(self, numberOfInputs, numberOfOutputs):
        layer = torch.nn.Sequential(
            torch.nn.Linear(numberOfInputs, numberOfOutputs),
            torch.nn.LeakyReLU(0.2)
        )
        return layer

    def act(self, observation, reward, done):
        inputTensor = torch.Tensor(self.RescaleObservation(observation))
        outputTensor = self.forward(inputTensor)
        highestNdx = outputTensor.argmax().item()
        return highestNdx


    def PerturbateWeights(self, layerNdx, weightsDeltaSigma, biasDeltaSigma):
        if layerNdx < 0 or layerNdx >= len(self.layers):
            raise ValueError("NNController.py PerturbateWeights(): The layer index ({}) is out of the range [0, {}]".format(layerNdx, len(self.layers) - 1))
        weightsShape = self.layers[layerNdx][0].weight.shape
        biasShape = self.layers[layerNdx][0].bias.shape

        weightsDelta = torch.randn(weightsShape) * weightsDeltaSigma
        biasDelta = torch.randn(biasShape) * biasDeltaSigma

        self.layers[layerNdx][0].weight = torch.nn.Parameter(
            self.layers[layerNdx][0].weight + weightsDelta
        )
        self.layers[layerNdx][0].bias = torch.nn.Parameter(
            self.layers[layerNdx][0].bias + biasDelta
        )

    def PerturbateAllWeights(self, weightsDeltaSigma, biasDeltaSigma):
        for layerNdx in range(len(self.layers)):
            self.PerturbateWeights(layerNdx, weightsDeltaSigma, biasDeltaSigma)

    def Save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))

    def RescaleObservation(self, observation):
        return [ (observation[0] - self.observation_low[0])/(self.observation_high[0] - self.observation_low[0]), \
                 (observation[1] - self.observation_low[1]) / (self.observation_high[1] - self.observation_low[1]), \
                 (observation[2] - self.observation_low[2]) / (self.observation_high[2] - self.observation_low[2]), \
                 (observation[3] - self.observation_low[3]) / (self.observation_high[3] - self.observation_low[3]), \
                 (observation[4] - self.observation_low[4]) / (self.observation_high[4] - self.observation_low[4]), \
                 (observation[5] - self.observation_low[5]) / (self.observation_high[5] - self.observation_low[5]) ]



def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.constant_(m.weight, 0)
            m.bias.data.fill_(0)


def TournamentStatistics(tournamentAverageRewards):
    if len(tournamentAverageRewards) == 0:
        raise ValueError("TournamentStatistics(): The input list is empty")
    highestReward = -sys.float_info.max
    rewardsSum = 0
    for reward in tournamentAverageRewards:
        if reward > highestReward:
            highestReward = reward
        rewardsSum += reward
    return rewardsSum / len(tournamentAverageRewards), highestReward



def main():
    print ("Acrobot-v1/NNController.py main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('OutputDirectory', help='The directory where the outputs will be written')
    parser.add_argument('--testController', help='The filepath of a neural network to test. Default: None',
                        default=None)
    parser.add_argument('--initialNeuralNet', help='The filepath for the starting neural network. Default: None', default=None)
    args = parser.parse_args()

    hiddenLayerWidthsList = (8,5)
    agent = NeuralNet(hiddenLayerWidthsList)
    if args.initialNeuralNet is not None:
        agent.Load(args.initialNeuralNet)

    if args.testController is not None:
        env = gym.make('Acrobot-v1')
        agent.Load(args.testController)
        rewardSumsList = []
        for i_episode in range(10):
            observation = env.reset()
            rewardSum = 0
            done = False
            while not done:
                env.render()
                print (observation)
                reward = 0
                done = False
                action = agent.act(observation, reward, done)
                #action = env.action_space.sample() # Random choice
                observation, reward, done, info = env.step(action)
                rewardSum += reward
                if done:
                    print ("Breaking! rewardSum = {}".format(rewardSum))
                    break
            rewardSumsList.append(rewardSum)
        print ("main(): rewardSumsList: {}".format(rewardSumsList))
        averageReward, highestReward = TournamentStatistics(rewardSumsList)
        print ("main(): averageReward = {}; highestReward = {}".format(averageReward, highestReward))
        sys.exit()

    """agent.PerturbateAllWeights(0.1, 0.1)
    inputsList = [-0.1, 0.4, -0.3, 0.7, 6.9, 15.0]
    inputsTensor = torch.Tensor(agent.RescaleObservation(inputsList))
    outputs = agent(inputsTensor)
    reward = 0
    done = False
    action = agent.act(inputsList, reward, done)
    print ("main(): outputs = {}; action = {}".format(outputs, action))
    """

    env = gym.make('Acrobot-v1')
    episode_count = 30
    reward = 0
    done = False

    numberOfPerturbations = 15
    weigtsDeltaSigma = 1.0
    biasDeltaSigma = 0.3

    highestReward = -sys.float_info.max
    currentChampion = copy.deepcopy(agent)
    numberOfTournaments = 100

    with open(os.path.join(args.OutputDirectory, 'stats.csv'), "w+") as statsFile:
        statsFile.write("Tournament,AverageReward,TournamentHighestReward,ChampionReward\n")

    for tournamentNdx in range(numberOfTournaments):
        print ("Tournament {}:".format(tournamentNdx + 1))
        theta1 = float(tournamentNdx) * 2 * math.pi / 10
        theta2 = float(tournamentNdx) * 2 * math.pi / 13
        tournamentAverageRewards = []
        for perturbationTrialNdx in range(numberOfPerturbations):
            perturbedAgent = copy.deepcopy(currentChampion)

            perturbedAgent.PerturbateAllWeights(weigtsDeltaSigma * math.cos(theta1),
                                                biasDeltaSigma * math.sin(theta2 + 0.5 * numpy.random.randn()))
            agentEpisodeReward = 0
            for episode in range(episode_count):
                observation = env.reset()
                rewardSum = 0
                done = False
                while not done:
                    action = perturbedAgent.act(observation, reward, done)  # Choose an action
                    # print ("action = {}".format(action))
                    observation, reward, done, info = env.step(action)  # Perform the action
                    rewardSum += reward
                    if done:
                        break
                    # env.render()
                    # Note there's no env.render() here. But the environment still can open window and
                    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                    # Video is not recorded every episode, see capped_cubic_video_schedule for details.
                agentEpisodeReward += rewardSum

            agentAverageEpisodeReward = agentEpisodeReward / episode_count
            print ("main(): agentAverageEpisodeReward = {}".format(agentAverageEpisodeReward))
            tournamentAverageRewards.append(agentAverageEpisodeReward)
            if agentAverageEpisodeReward > highestReward:
                highestReward = agentAverageEpisodeReward
                currentChampion = copy.deepcopy(perturbedAgent)
                currentChampion.Save(os.path.join(args.OutputDirectory, \
                                                  'champion_' + str(hiddenLayerWidthsList) + '_' + str(highestReward)))

        tournamentAverageReward, tournamentHighestReward = TournamentStatistics(tournamentAverageRewards)
        print ("highestReward = {}; tournamentAverageReward = {}; tournamentHighestReward = {}".format(highestReward,
                                                                                                       tournamentAverageReward,
                                                                                                       tournamentHighestReward))
        with open(os.path.join(args.OutputDirectory, 'stats.csv'), "a+") as statsFile:
            statsFile.write(str(tournamentNdx) + ',' + str(tournamentAverageReward) + ',' + \
                            str(tournamentHighestReward) + ',' + str(highestReward) + '\n')

    # Show the behaviour of the champion
    observation = env.reset()
    episodeSum = 0
    while True:
        action = currentChampion.act(observation, reward, done)  # Choose an action
        ob, reward, done, _ = env.step(action)  # Perform the action
        episodeSum += reward
        if done:
            break
        env.render()
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    print ("main(): Final champion, episodeSum = {}".format(episodeSum))

    # Close the env and write monitor result info to disk
    env.close

if __name__ == '__main__':
    main()