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

    expected = [[0],[1],[1],[0]]    # XOR
    #expected = [[1],[0],[0],[1]]   # XNOR
    #expected = [[0],[1],[1],[1]]   # OR
    #expected = [[0],[0],[0],[1]]   # AND
    #expected = [[0],[1],[1],[0]]

    for player in population.pop:

        player.think(vision=[0,0])
        decision = player.decision
        total= getScore(decision, expected[0])

        player.think(vision=[1,0])
        decision = player.decision
        total+= getScore(decision, expected[1])

        player.think(vision=[0,1])
        decision = player.decision
        total+= getScore(decision, expected[2])

        player.think(vision=[1,1])
        decision = player.decision
        total+= getScore(decision, expected[3])

        player.score = total
        continue

    bestMeep = population.bestMeeple
    print()
    bestMeep.think(vision=[0,0])
    decision = bestMeep.decision
    print( "Expected %s, Got %.4f, Score %f" % (expected[0], decision[0], getScore(decision, expected[0])))

    bestMeep.think(vision=[1,0])
    decision = bestMeep.decision
    print( "Expected %s, Got %.4f, Score %s" % (expected[1], decision[0], getScore(decision, expected[1])))

    bestMeep.think(vision=[0,1])
    decision = bestMeep.decision
    print( "Expected %s, Got %.4f, Score %s" % (expected[2], decision[0], getScore(decision, expected[2])))

    bestMeep.think(vision=[1,1])
    decision = bestMeep.decision
    print( "Expected %s, Got %.4f, Score %s" % (expected[3], decision[0], getScore(decision, expected[3])))






def getScore(decision:List[float], expected:List[float]):
    runningSum = 0
    for i in range(len(decision)):
        if ((decision[i] - expected[i])**2+1) != 0:
            runningSum += 1000/((decision[i] - expected[i])**2+1)
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

    for i in range(10):
        log.logger.warning("Testing round %d"%i)
        pop_i = i
        population = Population(100, 2, 1)
        #pyglet.clock.schedule_interval_soft(update, 1)
        pyglet.app.run()
        stats.append(population.generation)


    print("len: %s\n"%len(stats),           #len
          "min: %s\n"%np.min(stats),           #min
          "max: %s\n"%np.max(stats),           #max
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