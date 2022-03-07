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

windowMain = pyglet.window.Window(1200, 800)
windowMPL = pyglet.window.Window(1200,800)


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
from statistics import Statistics

from itertools import combinations
from itertools import permutations
from typing import List
from typing import Tuple


# [ x/bool, o/bool, 2:10 = board ]

statswindow = Statistics()

popcap = 1000
pop = Population(popcap, 11, 9)
curgame = 0
steps = 50
maxgames = popcap
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


    curgame = 0

    for dummy in range(steps//5):
        print("    -",end="")
    for dummy in range((steps%5)-1):
        print(" ", end="")
    print("|", maxgames, "/", gamestep, "/", steps)


    # Game Section Start
    matchviablemeepis:List[int] = list()

    for i in range(popcap):
        if cell_test(i):
            matchviablemeepis.append(i);

    if len(matchviablemeepis) >= 2: # can't match 1 or 0 meeps
        # [{players}, winner, turns]
        for i in range(1,8,2):
            for players in combinations((matchviablemeepis[:len(matchviablemeepis)//i]), 2):
                pre_game(players)
            matchviablemeepis.sort(key=lambda meepi: pop.pop[meepi].elo.rating, reverse=True)
    else:
        logger.warning("No match viable meeples");


    # Game Section End

    maxgames = curgame;
    gamestep = maxgames//steps
    if gamestep < steps:
        gamestep = steps

    print("")
    meep1:Meeple = pop.pop[0]
    meep2:Meeple = pop.pop[1]
    logger.info("Meep1: %.1f %.3f" %
                 (meep1.elo.rating, meep1.elo.uncertainty))
    logger.info("Meep2: %.1f %.3f" %
                 (meep2.elo.rating, meep2.elo.uncertainty))

    logger.info("Games took :%.2fs" % (time.time()-lasttime[0]));

    lasttime[1] = time.time()
    pop.naturalSelection()
    logger.info("NaS took :%.2fs" % (time.time()-lasttime[1]));
    logger.info("Gen took :%.2fs" % (time.time()-lasttime[0]));

    statswindow.update(pop.generation,
                       pop.genscoresHistor_max,
                       pop.genscoresHistor_cur,
                       pop.scorehistogHistor,
                       pop.speciesScoreHistogram);


@windowMain.event
def on_draw():
    windowMain.clear()
    pyglet.gl.glClearColor(0.7,0.7,0.7,1)
    pop.pop[0].brain.drawNetwork(50,50,1150,750)
    genlabel.text = "Generation: "+ str(pop.generation)
    genlabel.draw()

@windowMPL.event
def on_draw():
    windowMPL.clear();
    pyglet.gl.glClearColor(0.7,0.7,0.7,1)
    if statswindow.image is not None:
        statswindow.image.blit(0,0);





def cell_test(meepi:int) -> bool:

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    meep = pop.pop[meepi]

    #if rng.random() < .5:
    #    turn = "X"
    #else:
    #    turn = "O"
    for turnstep in range(8):
        meep.think(vision=[1,0]+board)
        decision = meep.decision
        index = decision.index(max(decision))
        if board[index] == 0:
            board[index] = 1
            meep.score += 1
        else:
            return False;

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]

    for turnstep in range(8):
        meep.think(vision=[0,1]+board)
        decision = meep.decision
        index = decision.index(max(decision))
        if board[index] == 0:
            board[index] = 2;
            meep.score += 1;
        else:
            return False;
    return True;


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
            meep1.score += 100
        elif endgamestate[2] == 2:
            meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 0)
            meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 1)
            meep2.score += 100
        #return endgamestate
    elif endgamestate[0] == "Draw":
        meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 0.25)
        meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 0.75)
        meep1.score += 25
        meep2.score += 75
        #return endgamestate



def run_game(meep1, meep2, show=False)->tuple:

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0];

    board[rng.integers(0,8)] = 1;
    # Force the first player to make a random move

    turn = "O"
    for turnstep in range(7):
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