from __future__ import annotations

import math
from itertools import combinations

import elo
from population import Population
import maintools
from maintools import rng
log = maintools.colLogger("blackjack")
from meeple import Meeple

from typing import Tuple

import gym



env:gym.Env = gym.make("Blackjack-v1");
env.reset();

def blackjackMain(population:Population):
    global env

    dealer_Elo = elo.Rating(1000);

    for meepi in range(population.size):
        meep:Meeple = population.meeples[meepi];


        for epi in range(50):
            observation:gym.Space = env.reset();
            while True:

                meep.think(vision=observation);
                decision = meep.decision

                # To hit or not to hit
                if decision[0] > .5:
                    observation, reward, done, info = env.step(action=1); #hit
                else:# decision[0] < .5:
                    observation, reward, done, info = env.step(action=0); #stand

                meep.score += reward

                if reward==1:
                    elo.rate_1v1(winner=meep.elo, loser=dealer_Elo, score=1);
                else:
                    elo.rate_1v1(loser=dealer_Elo, winner=meep.elo, score=1);

                if done:
                    break;
    log.logger.info("Dealer rating: %.1f %.3f"%(dealer_Elo.rating, dealer_Elo.uncertainty))