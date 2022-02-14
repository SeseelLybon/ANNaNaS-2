# Code stolen from, I mean inspired by CodeBullet, as per usual
import statistics

import colorlog
import logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
logger = colorlog.getLogger('XOR')
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

import numpy as np
from numpy.random import default_rng
rng = default_rng()

import pyglet
window = pyglet.window.Window(1200,800)
pyglet.gl.glClearColor(0.7,0.7,0.7,1)

from multiprocessing import Process
import math
from typing import List

#from pymunk import Vec2d

from population import Population
from meeple import Meeple

from itertools import combinations

popcap = 1000
pop:Population

steps = 20
maxgames = popcap
gamestep = maxgames//steps

pop_i = 0

genlabel = pyglet.text.Label('23423423',
                          font_name='Times New Roman',
                          font_size=20,
                          x=100, y=750,
                          anchor_x='left', anchor_y='center',
                          color=(0,0,0, 255))

poplabel = pyglet.text.Label('321321321',
                             font_name='Times New Roman',
                             font_size=20,
                             x=100, y=650,
                             anchor_x='left', anchor_y='center',
                             color=(0,0,0, 255))

def update(dt):
    global steps
    global pop
    global gamestep
    global window

    print("---------------------------------------")
    print("New generation: ", pop.generation)
    # each update is a generation

    curgame = 0

    for dummy in range(steps//5):
        print("    -",end="")
    for dummy in range((steps%5)-1):
        print(" ", end="")
    print("|", maxgames, "/", gamestep, "/", steps)
    curgame+=1

    expected = [[0],[1],[1],[0]]

    for player in pop.pop:
        curgame+=1
        if curgame%gamestep==0:
            print("=",end="")

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

    bestMeep:Meeple = None
    for meep in pop.pop:
        if bestMeep is None or meep.score > bestMeep.score:
            bestMeep = meep
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

    window.clear()
    bestMeep.brain.drawNetwork(50,50,1150,750)
    genlabel.text = "Generation: "+ str(pop.generation)
    genlabel.draw()
    poplabel.text = "Population:"+ str(pop_i)
    poplabel.draw()

    # bestMeep.brain.printNetwork()
    #if (bestMeep.score/400)>0.9999:
    if bestMeep.score >= 4000:
        logger.fatal("Meep solved problem")
        pyglet.clock.unschedule(update)
        pyglet.app.exit()

    if pop.generation >= 1000:
        logger.fatal("Took 1k genertions. Ending population.")
        pyglet.clock.unschedule(update)
        pyglet.app.exit()

    pop.naturalSelection()

def getScore(decision:List[float], expected:List[float]):
    runningSum = 0
    for i in range(len(decision)):
        runningSum += 1000/((decision[i] - expected[i])**2+1)
    return runningSum


if __name__ == "__main__":

    import timeit
    import cProfile
    print("Starting Main.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()
    stats = []

    for i in range(10):
        logger.warning("Testing round %d"%i)
        pop_i = i
        pop = Population(popcap, 2, 1)
        pyglet.clock.schedule_interval_soft(update, 1)
        pyglet.app.run()
        stats.append(pop.generation)


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

    print("Finished Main.py as main")