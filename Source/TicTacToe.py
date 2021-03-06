# Code stolen from, I mean inspired by CodeBullet, as per usual
import multiprocessing

import numpy as np

#import pandas
#import matplotlib
#import seaborn

import pyglet
import trueskill
trueskill.TrueSkill(backend='scipy')

from meeple import Meeple

window = pyglet.window.Window(1200,800)
pyglet.gl.glClearColor(0.7,0.7,0.7,1)

import colorlog
import logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
logger = colorlog.getLogger('XOR')
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

from numpy.random import default_rng
rng = default_rng()

from population import Population

from itertools import combinations
from typing import List

popcap = 100
pop = Population(popcap, 11, 9)

# [ x/bool, o/bool, 2:10 = board ]

curgame = 0
steps = 100
maxgames = sum(1 for dummy in combinations(pop.pop, 2))*2
gamestep = maxgames//steps

genlabel = pyglet.text.Label('23423423',
                             font_name='Times New Roman',
                             font_size=20,
                             x=100, y=750,
                             anchor_x='left', anchor_y='center',
                             color=(0,0,0, 255))

def update(dt):
    global curgame
    global steps
    global pop
    global gamestep

    print("---------------------------------------")
    print("New generation: ", pop.generation)
    # each update is a generation

    curgames = 0

    for dummy in range(steps//5):
        print("    -",end="")
    for dummy in range((steps%5)-1):
        print(" ", end="")
    print("|", maxgames, "/", gamestep, "/", steps)

    bestMatch = None
    # [{players}, winner, turns]

    for players in combinations(list(range(popcap)), 2):
        endgamestate = pre_game(players)

        if endgamestate[2] == 1: # if player 1/X won
            if bestMatch is None: # if there's no best match, this wins
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] == "Foul": # if the best and this match are a foul, the match with the most steps wins
                if bestMatch[1][1] < endgamestate[1]:
                    bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] == "Draw": # Draws are better than fouls. only first draw needs to be stored
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] ==  "Winner": # Wins are better than fouls
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Draw" and endgamestate[0] ==  "Winner": # Wins are better than fouls
                bestMatch = [players, endgamestate]

        endgamestate = pre_game( (players[1], players[0]) )

        if endgamestate[2] == 1: # if player 1/X won
            if bestMatch is None: # if there's no best match, this wins
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] == "Foul": # if the best and this match are a foul, the match with the most steps wins
                if bestMatch[1][1] < endgamestate[1]:
                    bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] == "Draw": # Draws are better than fouls. only first draw needs to be stored
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Foul" and endgamestate[0] ==  "Winner": # Wins are better than fouls
                bestMatch = [players, endgamestate]
            elif bestMatch[1][0] == "Draw" and endgamestate[0] ==  "Winner": # Wins are better than fouls
                bestMatch = [players, endgamestate]


    print("")

    meep1:Meeple = pop.pop[bestMatch[0][0]]
    meep2:Meeple = pop.pop[bestMatch[0][1]]

    print("")
    if bestMatch:
        print("Best match played by meeps:", bestMatch[0][0], "and", bestMatch[0][1])
        print("Meep1:", meep1.winx, meep1.wino, meep1.losex, meep1.loseo, meep1.drawx, meep1.drawo, meep1.foulx, meep1.foulo)
        print("Meep2:", meep2.winx, meep2.wino, meep2.losex, meep2.loseo, meep2.drawx, meep2.drawo, meep2.foulx, meep2.foulo)
        print("Ratings:", trueskill.expose(meep1.rating), "and", trueskill.expose(meep2.rating) )
        print("Quality:", trueskill.quality_1vs1(meep1.rating, meep2.rating))
        print(run_game(meep1, meep2, show=True))
        print("Best match played by meeps:", bestMatch[0][1], "and", bestMatch[0][0])
        print(run_game(meep1, pop.pop[bestMatch[0][0]], show=True))
    print("")

    window.clear()
    pop.pop[bestMatch[0][0]].brain.drawNetwork(50,50,1150,750)
    genlabel.text = "Generation: "+ str(pop.generation)
    genlabel.draw()

    pop.naturalSelection()



def pre_game(players):
    global curgame
    global pop
    curgame+=1
    if curgame%gamestep==0:
        print("=",end="")

    meep1:Meeple = pop.pop[players[0]]
    meep2:Meeple = pop.pop[players[1]]
    #meep1:Meeple = players[0]
    #meep2:Meeple = players[1]

    quality = trueskill.quality_1vs1(meep1.rating, meep2.rating)

    endgamestate:tuple = run_game(meep1, meep2)
    #endgamestate = tuple(["Foul", 1, 1])
    if endgamestate[0] == "Winner":
        if endgamestate[2] == 1:
            meep1.rating, meep2.rating = trueskill.rate_1vs1(meep1.rating, meep2.rating)
            meep1.score += 90*quality
            meep2.score += 60*quality
            meep1.winx +=1
            meep2.loseo +=1
        elif endgamestate[2] == 2:
            meep2.rating, meep1.rating = trueskill.rate_1vs1(meep2.rating, meep1.rating)
            meep1.score += 50*quality
            meep2.score += 80*quality
            meep1.losex +=1
            meep2.wino +=1
        return endgamestate
    if endgamestate[0] == "Draw":
        meep1.rating, meep2.rating = trueskill.rate_1vs1(meep1.rating, meep2.rating, drawn=True)
        meep1.score += 50*quality
        meep2.score += 60*quality
        meep1.drawx +=1
        meep2.drawo +=1
        return endgamestate
    elif endgamestate[0] == "Foul":
        if endgamestate[2] == 1: # 2 caused the foul so gains less points
            meep1.rating, meep2.rating = trueskill.rate_1vs1(meep1.rating, meep2.rating, drawn=True)
            meep1.score += max([min([endgamestate[1]-1, 1]), 0])*quality
            meep2.score += max([min([endgamestate[1]-2, 1]), 0])*quality
            meep1.foulx +=1
        elif endgamestate[2] == 2:
            meep2.rating, meep1.rating = trueskill.rate_1vs1(meep2.rating, meep1.rating, drawn=True)
            meep1.score += max([min([endgamestate[1]-2, 1]), 0])
            meep2.score += max([min([endgamestate[1]-1, 1]), 0])
            meep2.foulo +=1
        return endgamestate



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

    if board[0]+board[1]+board[2] == 3 or \
            board[3]+board[4]+board[5] == 3 or \
            board[6]+board[7]+board[8] == 3 or \
            board[0]+board[3]+board[6] == 3 or \
            board[1]+board[4]+board[7] == 3 or \
            board[2]+board[5]+board[8] == 3 or \
            board[0]+board[4]+board[8] == 3 or \
            board[6]+board[4]+board[2] == 3:
        return 1

    if board[0]+board[1]+board[2] == -3 or \
            board[3]+board[4]+board[5] == -3 or \
            board[6]+board[7]+board[8] == -3 or \
            board[0]+board[3]+board[6] == -3 or \
            board[1]+board[4]+board[7] == -3 or \
            board[2]+board[5]+board[8] == -3 or \
            board[0]+board[4]+board[8] == -3 or \
            board[6]+board[4]+board[2] == -3:
        return 2

    return 0


def getScore(decision:List[float], expected:List[float]):
    runningSum = 0
    for i in range(len(decision)):
        runningSum += 1000/((decision[i] - expected[i])**2+1)
    return runningSum


if __name__ == "__main__":

    import cProfile
    print("Starting TicTacToe.py as __main__")

    #in Terminal -> snakeviz source/profiledprofile.prof
    #cProfile.run('update(10)', filename="profiledprofile.prof")

    pyglet.clock.schedule_interval_soft(update, 1)
    pyglet.app.run()

    print("Finished TicTacToe.py as __main__")