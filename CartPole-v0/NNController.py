import argparse
import gym
import torch
from collections import OrderedDict
import copy
import os
import math
import numpy
import sys


class NeuralNet(torch.nn.Module):
    def __init__(self, hiddenLayerWidths=(8,8,4)):
        super(NeuralNet, self).__init__()
        layersDict = OrderedDict()
        for hiddenLayerNdx in range(len(hiddenLayerWidths) + 1):
            if hiddenLayerNdx == 0:
                numberOfInputs = 4
            else:
                numberOfInputs = hiddenLayerWidths[hiddenLayerNdx - 1]

            if hiddenLayerNdx == len(hiddenLayerWidths):
                numberOfOutputs = 2
            else:
                numberOfOutputs = hiddenLayerWidths[hiddenLayerNdx]
            layersDict['layer' + str(hiddenLayerNdx)] = self.FullyConnectedLayer(numberOfInputs, numberOfOutputs)
            #print ("__init__(): layer = {}".format(layersDict['layer' + str(hiddenLayerNdx)]))
        self.layers = torch.nn.Sequential(layersDict)
        self.apply(init_weights)


    def forward(self, inputs):
        dataState = inputs
        for layerNdx in range(len(self.layers)):
            #print ("forward(): self.layers[layerNdx] = {}".format(self.layers[layerNdx]))
            dataState = self.layers[layerNdx](dataState)
        return torch.nn.functional.softmax(dataState, dim=0)

    def FullyConnectedLayer(self, numberOfInputs, numberOfOutputs):
        layer = torch.nn.Sequential(
            torch.nn.Linear(numberOfInputs, numberOfOutputs),
            torch.nn.LeakyReLU(0.2)
        )
        return layer

    def act(self, observation, reward, done):
        #print ("act(): observation = {}".format(observation))
        inputTensor = torch.Tensor(observation)
        #print ("act(): inputTensor = {}".format(inputTensor))
        outputTensor = self.forward(inputTensor)
        #print ("act(): outputTensor = {}".format(outputTensor))
        highestNdx = outputTensor.argmax().item()
        #print ("act(): highestNdx = {}".format(highestNdx))
        return highestNdx

    def PerturbateWeights(self, layerNdx, weightsDeltaSigma, biasDeltaSigma):
        if layerNdx < 0 or layerNdx >= len(self.layers):
            raise ValueError("NNController.py PerturbateWeights(): The layer index ({}) is out of the range [0, {}]".format(layerNdx, len(self.layers) - 1))
        weightsShape = self.layers[layerNdx][0].weight.shape
        #print ("PerturbateWeights(): weightsShape = {}".format(weightsShape))
        biasShape = self.layers[layerNdx][0].bias.shape
        #print ("PerturbateWeights(): biasShape = {}".format(biasShape))

        #print ("PerturbateWeights(): Before, self.layers[layerNdx][0].weight = \n{}".format(self.layers[layerNdx][0].weight))
        #print ("PerturbateWeights(): Before, self.layers[layerNdx][0].bias = \n{}".format(
        #    self.layers[layerNdx][0].bias))
        weightsDelta = torch.randn(weightsShape) * weightsDeltaSigma
        biasDelta = torch.randn(biasShape) * biasDeltaSigma

        self.layers[layerNdx][0].weight = torch.nn.Parameter(
            self.layers[layerNdx][0].weight + weightsDelta
        )
        self.layers[layerNdx][0].bias = torch.nn.Parameter(
            self.layers[layerNdx][0].bias + biasDelta
        )
        #print ("PerturbateWeights(): After, self.layers[layerNdx][0].weight = \n{}".format(
        #    self.layers[layerNdx][0].weight))
        #print ("PerturbateWeights(): After, self.layers[layerNdx][0].bias = \n{}".format(
        #    self.layers[layerNdx][0].bias))

    def PerturbateAllWeights(self, weightsDeltaSigma, biasDeltaSigma):
        for layerNdx in range(len(self.layers)):
            self.PerturbateWeights(layerNdx, weightsDeltaSigma, biasDeltaSigma)

    def Save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))

def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.constant_(m.weight, 0)
            m.bias.data.fill_(0)

def TournamentStatistics(tournamentAverageRewards):
    if len(tournamentAverageRewards) == 0:
        raise ValueError("TournamentStatistics(): The input list is empty")
    highestReward = -1
    rewardsSum = 0
    for reward in tournamentAverageRewards:
        if reward > highestReward:
            highestReward = reward
        rewardsSum += reward
    return rewardsSum/len(tournamentAverageRewards), highestReward


def main():
    print ('NNController.py main()')
    parser = argparse.ArgumentParser()
    parser.add_argument('OutputDirectory', help='The directory where the outputs will be written')
    parser.add_argument('--testController', help='The filepath of a neural network to test. Default: None', default=None)
    args = parser.parse_args()

    agent = NeuralNet((8,4))

    if args.testController is not None:
        env = gym.make('CartPole-v0')
        agent.Load(args.testController)
        rewardSumsList = []
        for i_episode in range(20):
            observation = env.reset()
            rewardSum = 0
            #for t in range(100):
            done = False
            while not done:
                env.render()
                print (observation)
                reward = 0
                done = False
                action = agent.act(observation, reward, done)
                #action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                rewardSum += reward
                if done:
                    print ("Breaking! rewardSum = {}".format(rewardSum))
                    break
            rewardSumsList.append(rewardSum)
        print ("main(): rewardSumsList: {}".format(rewardSumsList))
        sys.exit()

    inputs = torch.Tensor([-0.1, 2.4, 0.7, -1.2])
    outputs = agent(inputs)
    print ("main(): outputs = {}".format(outputs))

    #agent.PerturbateWeights(0, 0.1, 0.1)

    env = gym.make('CartPole-v0')
    episode_count = 100
    reward = 0
    done = False

    numberOfPerturbations = 5
    weigtsDeltaSigma = 0.3
    biasDeltaSigma = 0.1

    highestReward = -1
    currentChampion = copy.deepcopy(agent)
    numberOfTournaments = 100

    with open(os.path.join(args.OutputDirectory, 'stats.csv'),"w+") as statsFile:
        statsFile.write("Tournament,AverageReward,TournamentHighestReward,ChampionReward\n")

    for tournamentNdx in range(numberOfTournaments):
        print ("Tournament {}:".format(tournamentNdx + 1))
        theta1 = float(tournamentNdx) * 2 * math.pi / 10
        theta2 = float(tournamentNdx) * 2 * math.pi / 13
        tournamentAverageRewards = []
        for perturbationTrialNdx in range(numberOfPerturbations):
            perturbedAgent = copy.deepcopy(currentChampion)

            perturbedAgent.PerturbateAllWeights(weigtsDeltaSigma * math.cos(theta1), biasDeltaSigma * math.sin(theta2 + 0.5 * numpy.random.randn()))
            agentEpisodeReward = 0
            for episode in range(episode_count):
                observation = env.reset()
                rewardSum = 0
                done = False
                while not done:
                    action = perturbedAgent.act(observation, reward, done) # Choose an action
                    observation, reward, done, _ = env.step(action) # Perform the action
                    rewardSum += reward
                    if done:
                        #print ("main() breaking! episodeSum = {}".format(episodeSum))
                        break
                    #env.render()
                    # Note there's no env.render() here. But the environment still can open window and
                    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                    # Video is not recorded every episode, see capped_cubic_video_schedule for details.
                #print ("main(): After episode {}, episodeSum = {}".format(episode, episodeSum))
                agentEpisodeReward += rewardSum

            agentAverageEpisodeReward = agentEpisodeReward / episode_count
            print ("main(): agentAverageEpisodeReward = {}".format(agentAverageEpisodeReward))
            tournamentAverageRewards.append(agentAverageEpisodeReward)
            if agentAverageEpisodeReward > highestReward:
                highestReward = agentAverageEpisodeReward
                currentChampion = copy.deepcopy(perturbedAgent)

        tournamentAverageReward, tournamentHighestReward = TournamentStatistics(tournamentAverageRewards)
        print ("highestReward = {}; tournamentAverageReward = {}; tournamentHighestReward = {}".format(highestReward, tournamentAverageReward, tournamentHighestReward))
        with open(os.path.join(args.OutputDirectory, 'stats.csv'), "a+") as statsFile:
            statsFile.write(str(tournamentNdx) + ',' + str(tournamentAverageReward) + ',' + \
                str(tournamentHighestReward) + ',' + str(highestReward) + '\n')

    # Save the champion
    currentChampion.Save(os.path.join(args.OutputDirectory, 'champion_8_4_' + str(highestReward)))

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
