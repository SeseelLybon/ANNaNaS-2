from __future__ import annotations

import math
from itertools import combinations


from population import Population
import maintools
from maintools import rng
log = maintools.colLogger("tictactoe")
from meeple import Meeple

from typing import Tuple

#library for taking screenshots quickily
#from mss import mss
#
#with mss() as sct:
#    sct.shot()

# Use "from this import *" ?


def tictactoeMain(population:Population):

    for meepi in range(population.size):
        maintools.loadingbar.loadingBarIncrement();
        for i in range(100000):
            if cell_test(meepi, population):
                continue;
            else:
                break;

        #while cell_test(meepi, population) and population.meeples[meepi].score <= (math.factorial(9)): # score 9!
        #    continue;

        while pre_game((population.meeples[rng.integers(0,population.size)], population.meeples[meepi])) and \
            pre_game((population.meeples[meepi], population.meeples[rng.integers(0,population.size)])):
                continue;

        #for i in range(1,8,2):
        #    for players in combinations(list(range(popcap//i)), 2):
        #        pre_game(players)
        #    pop.pop.sort(key=lambda meep: meep.elo.rating, reverse=True)




def cell_test2(meepi:int, population:Population) -> bool:
    meep = population.meeples[meepi]

    board = list(rng.integers(0,3, [9])); # create a random board
    while checkWinner(board): # check if it's a valid board (nobody won yet)
        board = list(rng.integers(0,3, [9])); # create a random board

    board[rng.integers(0,9)] = 0; # set a spot to 0 so it can always move
    board_free = [ 1 if c == 0 else 0 for c in board ];

    meep.think(vision=[1,0]+board+board_free)
    decision = meep.decision
    index = decision.index(max(decision))
    if board[index] == 0:
        meep.score += 1
        return True;
    else:
        return False;

celltestscoreX=[];
celltestscoreO=[];

def cell_test(meepi:int, population:Population) -> bool:
    meep = population.meeples[meepi]

    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    #board[rng.integers(0,9)] = 1;

    #if rng.random() < .5:
    #    turn = "X"
    #else:
    #    turn = "O"
    turn = "X";
    for turnstep in range(8):
        board_free = [ 1 if c == 0 else 0 for c in board ];

        meep.think(vision=[1,0]+board+board_free)
        decision = meep.decision
        index = decision.index(max(decision))
        if board[index] == 0:
            if turn=="X":
                board[index] = 1
                meep.score += turnstep;
            elif turn=="O":
                board[index] = 1
                meep.score += turnstep;
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
    winchance1 = elo.winChance(meep1.elo, meep2.elo);
    winchance2 = elo.winChance(meep2.elo, meep1.elo);
    if endgamestate[0] == "Winner":
        if endgamestate[2] == 1:
            meep1.elo.newRating(winchance1, 1)
            meep2.elo.newRating(winchance2, 0)
            meep1.score += 10**5*winchance2
        elif endgamestate[2] == 2:
            meep1.elo.newRating(winchance1, 0)
            meep2.elo.newRating(winchance2, 1)
            meep2.score += 10**6*winchance1
        return True
    elif endgamestate[0] == "Draw":
        meep1.elo.newRating(winchance1, 0.25)
        meep2.elo.newRating(winchance2, 0.75)
        meep1.score += 10**3*winchance2
        meep2.score += 10**4*winchance1
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
            board_free = [ 1 if c == 0 else 0 for c in board ];

            meep1.think(vision=[1,0]+board+board_free)
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
            board_free = [ 1 if c == 0 else 0 for c in board ];

            meep2.think(vision=[1,0]+board+board_free)
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






import unittest
class Test_tictactoe(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from logging import DEBUG
        log.logger.setLevel(DEBUG);

    def setUp(self)->None:
        pass;

    def tearDown(self)->None:
        pass;

    def test_checkWinner(self):
        boards_test = [ [0,0,0, 0,0,0, 0,0,0], # Nothing
                        [1,1,1, 0,0,0, 0,0,0], # Horizontal
                        [2,2,2, 0,0,0, 0,0,0],
                        [0,0,0, 1,1,1, 0,0,0],
                        [0,0,0, 2,2,2, 0,0,0],
                        [0,0,0, 0,0,0, 1,1,1],
                        [0,0,0, 0,0,0, 2,2,2],
                        [1,0,0, 1,0,0, 1,0,0], # Vertical
                        [2,0,0, 2,0,0, 2,0,0],
                        [0,1,0, 0,1,0, 0,1,0],
                        [0,2,0, 0,2,0, 0,2,0],
                        [0,0,1, 0,0,1, 0,0,1],
                        [0,0,2, 0,0,2, 0,0,2],
                        [1,0,0, 0,1,0, 0,0,1], #Diagnal
                        [2,0,0, 0,2,0, 0,0,2],
                        [0,0,1, 0,1,0, 1,0,0],
                        [0,0,2, 0,2,0, 2,0,0]];
        boards_answers = [0, 1,2,1,2,1,2 ,1,2,1,2,1,2 ,1,2 ];

        for t, a in zip(boards_test, boards_answers):
            self.assertTrue(checkWinner(t)==a);

    def cell_test3(self):
        pass;
    #@unittest.expectedFailure
    #def functioniexpecttofail(self):
    #   pass;

    #def someTest(self):
    #    with self.subTest("example test"):
    #        self.assertTrue(1==1);


