from __future__ import annotations

import math
from itertools import combinations
import numpy as np

from population import Population
import maintools
from maintools import rng
from maintools import loadingbar
log = maintools.colLogger("binarytodecimal")
from meeple import Meeple

from typing import Tuple

import gym
env:gym.Env = gym.make("Acrobot-v1");
env.reset();


def acrobatMain(population:Population):
    for meepi in range(population.size):
        loadingbar.loadingBarIncrement();
        meep:Meeple = population.meeples[meepi];


        observation:gym.Space = env.reset();
        meep.score = 500;
        while True:

            meep.think(vision=observation);
            decision = meep.decision;
            decisionIndex = decision.index(max(decision));
            observation, reward, done, info = env.step(action=decisionIndex); #push right

            meep.score += reward;

            if done:
                break;

def acrobatReplayBest(meep:Meeple):
    observation:gym.Space = env.reset();
    while True:

        meep.think(vision=observation);
        decision = meep.decision;
        decisionIndex = decision.index(max(decision));
        observation, reward, done, info = env.step(action=decisionIndex); #push right

        meep.score += reward;
        env.render();
        if done:
            break;

if __name__ == "__main__":
    from neuralnetwork import NeuralNetwork;

    brain = NeuralNetwork.JSONLoadNeuralNetwork(filepath="./BestMeepleBrain.json");
    meep = Meeple(brain.input_size, brain.output_size, isHollow=True);
    meep.brain = brain;

    acrobatReplayBest(meep);