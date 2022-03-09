from __future__ import annotations


# https://en.wikipedia.org/wiki/Elo_rating_system

from typing import List


import maintools
from maintools import rng
log = maintools.colLogger("tictactoe")

from itertools import combinations
from numpy.random import default_rng

class ELO:
    Kfactor = 32

def rate_1v1(winner:Rating, loser:Rating, score:float, isDraw=False):
    winchance = winChance(winner, loser);
    if not isDraw:
        winner.newRating( winchance, score)
        loser.newRating( 1-winchance, 1-score)
    else:
        winner.newRating( winchance, 0.5)
        loser.newRating( winchance, 0.5)
    return;

# Probability of playerA winning over PlayerB
def winChance(Ra:Rating, Rb:Rating):
    step1 = Rb.rating-Ra.rating
    step2 = step1/400
    step3 = 10**step2
    step4 = 1+step3
    return 1/step4

class Rating:

    def __init__(self, rating:int=1000):
        self.truerating_debug = rng.normal();
        self.wins:int = 0;
        self.loses:int = 0;
        self.draws:int = 0;

        self.uncertainty:float = 1;

        self.rating:float = rating
        return;

    def __repr__(self):
        return str({"rating":round(self.rating,2),
                    "uncertainty":round(self.uncertainty,2),
                    "wins":self.wins,
                    "loses":self.loses,
                    "truerating":self.truerating_debug})

    def newRating(self, winchance:float, winscore:float, isDraw=False):
        if isDraw:
            self.draws+=1;
        elif winscore > 0.5:
            self.wins+=1;
        else:
            self.loses+=1;

        #chance self wins over playerB
        #winchance = winChance(self, playerB);

        self.rating = self.rating+(ELO.Kfactor*self.uncertainty)* (winscore - winchance)
        #self.rating = self.rating+ELO.Kfactor* (winscore - winchance);
        self.newUncertainty(winchance, winscore)

    def newUncertainty(self, winchance:float, winscore:float):
        """Uncertainty

        Prototype variable for later
        Uncertainty indicates how uncertain I am that rating is the correct score.
        Lowers impact of K-factor.
        Performing as expected lowers it, unexpected results increase it.
        So a player with 1500 and uncertainty 1 is new,
        where as a 1500 player with uncerainty 0.1 has been playing for a while and just happens to rest at the entrance ELO
        """
        newuncertainty = self.uncertainty
        # if prediction is correct, lower uncertainty
        if (winchance < 0.5 and winscore < 0.5) or (winchance > 0.5 and winscore > 0.5):
            newuncertainty *= 0.99
            # if very correct, raise a bit more
        else:
            newuncertainty *= 1.01
        # if prediction is incorrect, raise uncertainty
            # if very incorrect, raise a bit more
        self.uncertainty = newuncertainty







import unittest
class Test_tictactoe(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from logging import DEBUG
        log.logger.setLevel(DEBUG);
        from numpy.random import default_rng
        cls.rng = default_rng(11036)

    def test_NewRating(self):
        winscores = list(rng.random([10]))
        for winscore in winscores:
            with self.subTest("%.2f"%winscore):
                playerA = Rating(1200);
                playerB = Rating(1000);
                winchance = winChance(playerA, playerB);
                playerA.newRating(winchance, winscore);
                playerB.newRating((1-winchance), (1-winscore));
                #log.logger.debug("%.2f %.2f - PlayerA %s"%(winscore, winchance, playerA.rating))
                #log.logger.debug("%.2f %.2f - PlayerB %s"%(1-winscore, 1-winchance, playerB.rating))
                lhs = round(playerA.rating,5)
                rhs = round(1200+(winscore-winchance)*32,5)
                self.assertTrue(lhs==rhs,
                                msg="Got:%s - Expected:%s"%(lhs, rhs));
                lhs = round(playerB.rating,5)
                rhs = round(1000+((1-winscore)-(1-winchance))*32,5)
                self.assertTrue(lhs==rhs,
                                msg="Got:%s - Expected:%s"%(lhs, rhs));

    def test_NewUncertainty(self):
        #winscores = [1,.75,.5,.25,0];
        winscores = list(rng.random([10]))
        for winscore in range(len(winscores)):
            winscore = winscores[winscore]
            with self.subTest("%.2f"%winscore):
                playerA = Rating(1200);
                playerB = Rating(1000);
                winchance = winChance(playerA, playerB);
                playerA.newRating(winchance, winscore);
                playerB.newRating((1-winchance), (1-winscore));

                lhs = round(playerA.uncertainty,5)
                rhs = round( 0.99 if (winchance < 0.5 and winscore < 0.5) or (winchance > 0.5 and winscore > 0.5) else 1.01, 5)
                self.assertTrue(lhs==rhs,
                                msg="Got:%s - Expected:%s"%(lhs, rhs));
                lhs = round(playerB.uncertainty,5)
                rhs = round( 0.99 if (winchance > 0.5 and winscore > 0.5) or (winchance < 0.5 and winscore < 0.5) else 1.01, 5)
                self.assertTrue(lhs==rhs,
                                msg="Got:%s - Expected:%s"%(lhs, rhs));



    #@unittest.expectedFailure
    #def functioniexpecttofail(self):
    #   pass;

    #def someTest(self):
    #    with self.subTest("example test"):
    #        self.assertTrue(1==1);

    def test_overlylargetestcase(self):
        players:List[Rating] = [Rating() for dummy in range(10)]

        players.sort(key=lambda p: p.truerating_debug, reverse=True);
        for i in range(len(players)):
            log.logger.debug("%s %s"%(i, players[i]));
        log.logger.debug("");

        #matches = list(combinations(players,2))


        for dummy in range(2000):
            #print("======")
            #matchi:int = self.rng.integers(0, len(matches))

            player1:Rating = players[self.rng.integers(0, len(players))]
            player2:Rating = players[self.rng.integers(0, len(players))]

            winchance1 = winChance(player1, player2)
            winchance2 = winChance(player2, player1)

            #skill1 = self.rng.integers(1,20)+player1.truerating_debug;
            #skill2 = self.rng.integers(1,20)+player2.truerating_debug;

            skill1 = player1.truerating_debug;
            skill2 = player2.truerating_debug;
            if skill1 > skill2:
                #print("player 1 won")
                player1.newRating(winchance1, 1)
                player1.newRating(winchance2, 0)
                #player1.newUncertainty(winchance1, 1)
                #player2.newUncertainty(winchance2, 0)
            else:
                #print("player 2 won")
                player1.newRating(winchance1, 0)
                player2.newRating(winchance2, 1)
                #player1.newUncertainty(winchance1, 0)
                #player2.newUncertainty(winchance2, 1)


        for i in range(len(players)):
            log.logger.debug("%s %s"%(i, players[i]));

