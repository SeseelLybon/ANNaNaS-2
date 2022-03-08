from __future__ import annotations



from population import Population
import maintools
from maintools import rng
log = maintools.colLogger("tictactoe")
from meeple import Meeple

from typing import Tuple


def TicTacToeMain(population:Population):

    for meepi in range(population.size):
        maintools.loadingbar.loadingBarIncrement();
        while cell_test2(meepi, population) and population.pop[meepi].score < 1*10**5: # score < 3 times 9!
            continue;
        while pre_game((population.bestMeeple, population.pop[meepi])) and pre_game((population.pop[meepi], population.bestMeeple)):
            continue;


def cell_test2(meepi:int, population:Population) -> bool:
    meep = population.pop[meepi]

    board = list(rng.integers(0,3, [9])); # create a random board
    while checkWinner(board): # check if it's a valid board (nobody won yet)
        board = list(rng.integers(0,3, [9])); # create a random board

    board[rng.integers(0,9)] = 0; # set a spot to 0 so it can always move

    meep.think(vision=[1,0]+board)
    decision = meep.decision
    index = decision.index(max(decision))
    if board[index] == 0:
        meep.score += 1
        return True;
    else:
        return False;


def cell_test(meepi:int, population:Population) -> bool:
    meep = population.pop[meepi]

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    board[rng.integers(0,9)] = 1;

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
    board[rng.integers(0,9)] = 1;

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

import elo

def pre_game(players:Tuple[Meeple,Meeple]):

    meep1:Meeple = players[0]
    meep2:Meeple = players[1]
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
        return True
    elif endgamestate[0] == "Draw":
        meep1.elo.newRating(elo.winChance(meep1.elo, meep2.elo), 0.25)
        meep2.elo.newRating(elo.winChance(meep2.elo, meep1.elo), 0.75)
        meep1.score += 25
        meep2.score += 75
        return True
    return False


def run_game(meep1:Meeple, meep2:Meeple, show=False)->tuple:
    # Vision
    # [ x:bool, o:bool, board ]

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0];

    board[rng.integers(0,9)] = 1;
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

        whowon = checkWinner(board)
        if whowon == 1:
            return tuple(["Winner", turnstep, 1])
        elif whowon == 2:
            return tuple(["Winner", turnstep, 2])

    return tuple(["Draw", 9, 0])



def checkWinner(board)->int:
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


if __name__ == "__main__":
    print("Start of TicTacToe")





    print("End of TicTacToe")