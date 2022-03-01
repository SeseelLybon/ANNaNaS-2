# Code stolen from, I mean inspired by CodeBullet, as per usual
import multiprocessing
import statistics
import time

import numpy as np

#import pandas
#import matplotlib
#import seaborn

import elo

import pyglet

from meeple import Meeple

window = pyglet.window.Window(1200,800)
pyglet.gl.glClearColor(0.7,0.7,0.7,1)

import colorlog
import logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
logger = colorlog.getLogger('TicTacToe')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from numpy.random import default_rng
rng = default_rng()

from population import Population

from itertools import combinations
from itertools import permutations
from typing import List
from typing import Tuple

popcap = 1000
pop = Population(popcap, 11, 9)

# [ x/bool, o/bool, 2:10 = board ]

curgame = 0
steps = 100
maxgames = sum(1 for dummy in combinations(pop.pop, 2))
#tempmg=0
#for i in range(1,2,2):
#    tempmg += maxgames//i
#maxgames=tempmg
print(maxgames)
gamestep = maxgames//steps

genlabel = pyglet.text.Label('23423423',
                             font_name='Times New Roman',
                             font_size=20,
                             x=100, y=750,
                             anchor_x='left', anchor_y='center',
                             color=(0,0,0, 255))

def update(dt):
    global curgame, steps, pop, gamestep, maxgames

    logger.info("---------------------------------------")
    logger.info("New generation: %d" % pop.generation)
    # each update is a generation
    temptime = time.time()
    lasttime = [temptime for i in range(2)]

    window.clear()
    pop.pop[0].brain.drawNetwork(50,50,1150,750)
    genlabel.text = "Generation: "+ str(pop.generation)
    genlabel.draw()

    curgame = 0

    for dummy in range(steps//5):
        print("    -",end="")
    for dummy in range((steps%5)-1):
        print(" ", end="")
    print("|", maxgames, "/", gamestep, "/", steps)

    # [{players}, winner, turns]
    for i in range(1,8,2):
        for players in combinations(list(range(popcap//i)), 2):
            pre_game(players)
        pop.pop.sort(key=lambda meep: meep.elo.rating, reverse=True)

    print("")
    meep1:Meeple = pop.pop[0]
    meep2:Meeple = pop.pop[1]
    logger.info("Meep1: %f %d %d %d %d %d %d %d %d" %
                 (meep1.elo.rating, meep1.winx, meep1.wino, meep1.losex, meep1.loseo, meep1.drawx, meep1.drawo, meep1.foulx, meep1.foulo))
    logger.info("Meep2: %f %d %d %d %d %d %d %d %d" %
                 (meep2.elo.rating, meep2.winx, meep2.wino, meep2.losex, meep2.loseo, meep2.drawx, meep2.drawo, meep2.foulx, meep2.foulo))

    logger.info("Games took :%.2fs" % (time.time()-lasttime[0]));

    lasttime[1] = time.time()
    pop.naturalSelection()
    maxgames = curgame
    gamestep = maxgames//steps
    logger.info("NaS took :%.2fs" % (time.time()-lasttime[1]));
    logger.info("Gen took :%.2fs" % (time.time()-lasttime[0]));



def pre_game(players):
    global curgame, pop
    curgame+=1
    if curgame%gamestep==0:
        print("=",end="")

    meep1:Meeple = pop.pop[players[0]]
    meep2:Meeple = pop.pop[players[1]]
    #meep1:Meeple = players[0]
    #meep2:Meeple = players[1]


    endgamestate:tuple = run_game(meep1, meep2)
    #endgamestate = tuple(["Foul", 1, 1])
    if endgamestate[0] == "Winner":
        if endgamestate[2] == 1:
            meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 1)
            meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 0)
            meep1.score += 90
            meep2.score += 60
            meep1.winx +=1
            meep2.loseo +=1
        elif endgamestate[2] == 2:
            meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 0)
            meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 1)
            meep1.score += 50
            meep2.score += 80
            meep1.losex +=1
            meep2.wino +=1
        #return endgamestate
    elif endgamestate[0] == "Draw":
        meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 0.25)
        meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 0.75)
        meep1.score += 50
        meep2.score += 60
        meep1.drawx +=1
        meep2.drawo +=1
        #return endgamestate

    #meep1.score = meep1.rating.rating
    #meep2.score = meep1.rating.rating
    elif endgamestate[0] == "Foul":
        if endgamestate[2] == 1: # 2 caused the foul so gains less points
            #meep2.elo.rating = ELO.newRating(meep1.elo.rating, ELO.winChance(meep1.elo.rating, meep2.elo.rating), 0.10)
            meep1.score += max([min([endgamestate[1]-1, 1]), 0])
            meep2.score += max([min([endgamestate[1]-2, 1]), 0])
            meep1.foulx += 1
        elif endgamestate[2] == 2:
            #meep1.elo.rating = ELO.newRating(meep2.elo.rating, ELO.winChance(meep2.elo.rating, meep1.elo.rating), 0.10)
            meep1.score += max([min([endgamestate[1]-2, 1]), 0])
            meep2.score += max([min([endgamestate[1]-1, 1]), 0])
            meep2.foulo +=1
        #return endgamestate
    #meep1.score = meep1.elo.rating;
    #meep2.score = meep2.elo.rating;



def run_game(meep1, meep2, show=False)->tuple:

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]

    #if rng.random() < .5:
    #    turn = "X"
    #else:
    #    turn = "O"
    turn = "X"
    for turnstep in range(8):
        if turn == "X":
            meep1.think(vision=[1,0]+board)
            decision = meep1.decision
            index = decision.index(max(decision))
            if show:
                print("X", decision)
            if board[index] == 0:
                board[index] = 1
            else:
                return tuple(["Foul", turnstep, 2]) # 2 wins
            turn = "O"
        elif turn == "O":
            meep2.think(vision=[0,1]+board)
            decision = meep2.decision
            index = decision.index(max(decision))
            if show:
                print("O", decision)
            if board[index] == 0:
                board[index] = -1
            else:
                return tuple(["Foul", turnstep, 1]) # 1 wins
            turn = "X"

        if show:
            print(board[0], board[1], board[2])
            print(board[3], board[4], board[5])
            print(board[6], board[7], board[8])

        whowon = checkWinner(board, meep1, meep2)
        if whowon == 1:
            return tuple(["Winner", turnstep, 1])
        elif whowon == 2:
            return tuple(["Winner", turnstep, 2])

    return tuple(["Draw", 9, 0])


def checkWinner(board, meep1, meep2)->int:
    winner = 0
    for dummy in range(1):
        #test columns
        if len({board[0], board[1], board[2]}) == 1:
            if board[0] != 0:
                return board[0]
        if len({board[3], board[4], board[5]}) == 1:
            if board[3] != 0:
                return board[3]
        if len({board[6], board[7], board[8]}) == 1:
            if board[6] != 0:
                return board[6]
        #test rows
        if len({board[0], board[3], board[6]}) == 1:
            if board[0] != 0:
                return board[0]
        if len({board[1], board[4], board[7]}) == 1:
            if board[1] != 0:
                return board[1]
        if len({board[2], board[5], board[8]}) == 1:
            if board[2] != 0:
                return board[2]
        #test diagnals
        if len({board[0], board[4], board[8]}) == 1:
            if board[4] != 0:
                return board[4]
        if len({board[6], board[4], board[2]}) == 1:
            if board[4] != 0:
                return board[4]
    return 0


#def getScore(decision:List[float], expected:List[float]):
#    runningSum = 0
#    for i in range(len(decision)):
#        runningSum += 1000/((decision[i] - expected[i])**2+1)
#    return runningSum


if __name__ == "__main__":

    import cProfile
    print("Starting TicTacToe.py as __main__")

    #in Terminal -> snakeviz source/profiledprofile.prof
    #cProfile.run('update(10)', filename="profiledprofile.prof")

    pyglet.clock.schedule_interval_soft(update, 1)
    pyglet.app.run()

    print("Finished TicTacToe.py as __main__")