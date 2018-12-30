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
>>> cartPoleEnv = gym.make('CartPole-v0')
(...)
>>> cartPoleEnv.action_space
Discrete(2)
>>> cartPoleEnv.observation_space
Box(4,)
>>> cartPoleEnv.observation_space.low
array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],
      dtype=float32)
>>> cartPoleEnv.observation_space.high
array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38],
      dtype=float32)

"""

class NeuralNet(torch.nn.Module):
    def __init__(self, hiddenLayerWidths=(8,5)):
        super(NeuralNet, self).__init__()
        layersDict = OrderedDict()
        for hiddenLayerNdx in range(len(hiddenLayerWidths) + 1):
            if hiddenLayerNdx == 0:
                numberOfInputs = 4 # cartPoleEnv.observation_space
            else:
                numberOfInputs = hiddenLayerWidths[hiddenLayerNdx - 1]

            if hiddenLayerNdx == len(hiddenLayerWidths):
                numberOfOutputs = 2 # cartPoleEnv.action_space, discrete
            else:
                numberOfOutputs = hiddenLayerWidths[hiddenLayerNdx]
            layersDict['layer' + str(hiddenLayerNdx)] = self.FullyConnectedLayer(numberOfInputs, numberOfOutputs)

        self.layers = torch.nn.Sequential(layersDict)
        self.apply(init_weights)
        self.observation_low = [-4.8000002e+00, 0., -4.1887903e-01, 0.] # No rescaling for infinite ranges
        self.observation_high = [4.8000002e+00, 1., 4.1887903e-01, 1.]


    def forward(self, inputs):
        dataState = inputs
        for layerNdx in range(len(self.layers)):
            dataState = self.layers[layerNdx](dataState)
            return torch.nn.functional.softmax(dataState, dim=0)

    def FullyConnectedLayer(self, numberOfInputs, numberOfOutputs):
        layer = torch.nn.Sequential(
            torch.nn.Linear(numberOfInputs, numberOfOutputs),
            torch.nn.ReLU()
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
                ]

    def MoveWeights(self, weightsDeltaList, biasDeltaList):
        if len(weightsDeltaList) != len(self.layers) or len (biasDeltaList) != len(self.layers):
            raise ValueError("MoveWeights(): The length weightsDeltaList({}) or the length of biasDeltaList ({}) doesn't equal the number of layers ({})".format(len(weightsDeltaList), len(biasDeltaList), len(self.layers)))
        for layerNdx in range(len(self.layers)):
            if weightsDeltaList[layerNdx].shape != self.layers[layerNdx][0].weight.shape:
                raise ValueError("MoveWeights(): At index {}, the shape of the weightsDelta ({}) doesn't match the shape of the layer weights ({})".format(layerNdx, weightsDeltaList[layerNdx].shape, self.layers[layerNdx][0].weight.shape))
            self.layers[layerNdx][0].weight = torch.nn.Parameter(
                self.layers[layerNdx][0].weight + weightsDeltaList[layerNdx]
            )

            if biasDeltaList[layerNdx].shape != self.layers[layerNdx][0].bias.shape:
                raise ValueError("MoveWeights(): At index {}, the shape of the biasDelta ({}) doesn't match the shape of the layer bias ({})".format(
                        layerNdx, biasDeltaList[layerNdx].shape, self.layers[layerNdx][0].bias.shape))
            self.layers[layerNdx][0].bias = torch.nn.Parameter(
                self.layers[layerNdx][0].bias + biasDeltaList[layerNdx]
            )


def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.xavier_uniform_(m.bias)


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
    print ("*** CartPole-v0/imitation.py main() ***\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('OutputDirectory', help='The directory where the outputs will be written')
    parser.add_argument('--testController', help='The filepath of a neural network to test. Default: None',
                        default=None)
    args = parser.parse_args()




if __name__ == '__main__':
    main()