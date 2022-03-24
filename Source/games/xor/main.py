# Code stolen from, I mean inspired by CodeBullet, as per usual
import statistics


import numpy as np
from numpy.random import default_rng
rng = default_rng()

import pyglet

from typing import List

from population import Population
import maintools
log = maintools.colLogger("xor")

def xorMain(population:Population):

    #maintools.loadingbar.loadingBarIncrement()


    test = [[0,0],[1,0],[0,1],[1,1]];

    #answer = [[0],[1],[1],[0]]    # XOR
    #answer = [[1],[0],[0],[1]]   # XNOR
    #answer = [[0],[1],[1],[1]]   # OR
    #answer = [[1],[0],[0],[0]]   # NOR
    answer = [[1],[1],[1],[0]]   # AND

    for player in population.meeples:
        total = 0;
        #index = np.array([0 for _ in range(len(test))]);
        #rng.shuffle(index);

        #for i in index:
        for t, a in zip(test, answer):
            player.think(vision=t)
            decision = player.decision
            total += getScore(decision, a)

        #player.brain.train([[0,0],[1,0],[0,1],[1,1]],expected,0.00001);

        player.score = total
        continue

    bestMeep = population.bestMeeple
    print()
    for t, a in zip(test, answer):
        bestMeep.think(vision=t)
        decision = bestMeep.decision
        print( "Expected %s, Got %.4f, Score %f" % (a, decision[0], getScore(decision, a)))



#def getScore(decision:List[float], expected:List[float]):
#    runningSum = 0
#    for x in range(len(decision)):
#        temp = -(decision[x] - expected[x])**3.2+400;
#        if temp < 0:
#            runningSum += 0;
#        else:
#            runningSum += temp;
#    return runningSum


def getScore(decision:List[float], expected:List[float]):
    runningSum = 0
    for i in range(len(decision)):
        if ((decision[i] - expected[i])**2) != 0:
            runningSum += 10000/((decision[i] - expected[i])**2+1)
        else:
            runningSum += 10;
    return runningSum





if __name__ == "__main__":

    import timeit
    import cProfile
    print("Starting xor.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()
    stats = []

    for _i in range(10):
        log.logger.warning("Testing round %d"%_i)
        pop_i = _i
        population = Population(100, 2, 1)
        #pyglet.clock.schedule_interval_soft(update, 1)
        pyglet.app.run()
        stats.append(population.generation)


    print("len: %s\n"%len(stats),           #len
          "min: %s\n"%min(stats),           #min
          "max: %s\n"%max(stats),           #max
          "mean: %s\n"%statistics.mean(stats),
          "median: %s\n"%statistics.median(stats),
          "mode: %s\n"%statistics.mode(stats),
          "quantiles: %s\n"%statistics.quantiles(stats),
          "pvariance: %s\n"%statistics.pvariance(stats)
          )


    #meep1 = Meeple(9,9)
    #meep2 = Meeple(9,9)
    #pre_game(tuple([meep1,meep2]))
    #pass

    #for players in combinations(pop.pop, 2):
    #    print(players[0], players[1])

    print("Finished xor.py as main")