# Code stolen from, I mean inspired by CodeBullet, as per usual
import multiprocessing

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
pop = Population(popcap, 2, 1)

# [ x/bool, o/bool, 2:10 = board ]

curgame = 0
steps = 20
maxgames = len(pop.pop)
gamestep = maxgames//steps


genlabel = pyglet.text.Label('23423423',
                          font_name='Times New Roman',
                          font_size=20,
                          x=100, y=750,
                          anchor_x='center', anchor_y='center',
                          color=(0,0,0, 255))

def update(dt):
    global curgame
    global steps
    global pop
    global gamestep
    global window

    print("---------------------------------------")
    print("New generation: ", pop.generation)
    # each update is a generation

    curgames = 0

    for dummy in range(steps//5):
        print("    -",end="")
    for dummy in range((steps%5)-1):
        print(" ", end="")
    print("|", maxgames, "/", gamestep, "/", steps)

    for player in pop.pop:

        player.think(vision=[0,0])
        decision = player.decision
        total= 1/((decision[0] - 0)**2+1)

        player.think(vision=[1,0])
        decision = player.decision
        total+= 1/((decision[0] - 1)**2+1)

        player.think(vision=[0,1])
        decision = player.decision
        total+= 1/((decision[0] - 1)**2+1)

        player.think(vision=[1,1])
        decision = player.decision
        total+= 1/((decision[0] - 0)**2+1)

        player.score = 1/total
        continue

    bestMeep:Meeple = None
    for meep in pop.pop:
        if bestMeep is None or meep.score > bestMeep.score:
            bestMeep = meep

    bestMeep.think(vision=[0,0])
    decision = bestMeep.decision
    print( decision, 1/((decision[0] - 0)**2+1))

    bestMeep.think(vision=[1,0])
    decision = bestMeep.decision
    print( decision, 1/((decision[0] - 1)**2+1))

    bestMeep.think(vision=[0,1])
    decision = bestMeep.decision
    print( decision, 1/((decision[0] - 1)**2+1))

    bestMeep.think(vision=[1,1])
    decision = bestMeep.decision
    print( decision, 1/((decision[0] - 1)**2+1))

    window.clear()
    bestMeep.brain.drawNetwork(50,50,1150,750)
    genlabel.text = "Generation: "+ str(pop.generation)
    genlabel.draw()

    bestMeep.brain.printNetwork()

    pop.naturalSelection()


if __name__ == "__main__":

    import timeit
    import cProfile
    print("Starting Main.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()

    pyglet.clock.schedule_interval_soft(update, 5)
    pyglet.app.run()

    #meep1 = Meeple(9,9)
    #meep2 = Meeple(9,9)
    #pre_game(tuple([meep1,meep2]))
    #pass

    #for players in combinations(pop.pop, 2):
    #    print(players[0], players[1])

    print("Finished Main.py as main")