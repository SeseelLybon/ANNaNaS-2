from __future__ import annotations

import math
from itertools import combinations


from population import Population
import maintools
from maintools import rng
log = maintools.colLogger("tictactoe")
from meeple import Meeple

from typing import Tuple

import gym



env:gym.Env = gym.make("HotterColder-v0");
env.reset();


def hottercolderMain(population:Population):
    global env

    #for meepi in range(population.size):
    #    observation:gym.Space = env.reset();
    #    done = False;
    #    while True:
    #        env.action_space.
#
    #        observation, reward, done, info = env.step();
    #        if done:
    #            break;
#
#
    #        pass;