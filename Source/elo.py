from __future__ import annotations


# https://en.wikipedia.org/wiki/Elo_rating_system

from typing import List

from itertools import combinations
from numpy.random import default_rng
rng = default_rng(11037)

class ELO:
    Kfactor = 32

def rate_1v1(winner:Rating, loser:Rating, score:float, isDraw=False):
    winner.newRating( winChance(winner, loser), score)
    loser.newRating( winChance(loser, winner), 1-score)
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
        self.wins = 0;
        self.loses = 0;

        self.uncertainty = 1;

        self.rating = rating
        return;

    def __repr__(self):
        return str({"rating":round(self.rating,2),
                    "uncertainty":round(self.uncertainty,2),
                    "wins":self.wins,
                    "loses":self.loses})

    def newRating(self, winchance:float, winscore:float):
        if winscore > 0.5:
            self.wins+=1;
        else:
            self.loses+=1;
        #self.rating = self.rating+(ELO.Kfactor*self.uncertainty)* (winscore - winchance)
        self.rating = self.rating+ELO.Kfactor* (winscore - winchance)
        self.newUncertainty(winchance, winscore)

    def newUncertainty(self, winchance:float, winscore:float):
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

    '''Uncertainty
    Prototype variable for later
    Uncertainty indicates how uncertain I am that rating is the correct score.
    Lowers impact of K-factor.
    Performing as expected lowers it, unexpected results increase it.
    So a player with 1500 and uncertainty 1 is new,
    where as a 1500 player with uncerainty 0.1 has been playing for a while and just happens to rest at the entrance ELO
    '''


if __name__ == "__main__":

    players:List[Rating] = [Rating() for dummy in range(10)]

    for i in range(len(players)):
        print(i, players[i]);
    print("")

    matches = list(combinations(players,2))


    for dummy in range(5000):
        #print("======")
        matchi:int = rng.integers(0, len(matches))

        player1:Rating = matches[matchi][0]
        player2:Rating = matches[matchi][1]

        winchance1 = winChance(player1, player2)
        winchance2 = winChance(player2, player1)
        #print("ratings: %f, %f"%(player1.rating, player2.rating))
        #print("winchances: %.2f, %.2f"%(winchance1,winchance2))
        #print("uncertainty: %.2f, %.2f"%(player1.uncertainty, player2.uncertainty))


        skill1 = rng.integers(1,20)+player1.trueskill_debug;
        skill2 = rng.integers(1,20)+player2.trueskill_debug;
        #print("Skill: %.2f, %.2f"%(skill1, skill2))
        if skill1 > skill2:
            #print("player 1 won")
            player1.newRating(winchance1, 1)
            player1.newRating(winchance2, 0)
            player1.newUncertainty(winchance1, 1)
            player2.newUncertainty(winchance2, 0)
        else:
            #print("player 2 won")
            player1.newRating(winchance1, 0)
            player2.newRating(winchance2, 1)
            player1.newUncertainty(winchance1, 0)
            player2.newUncertainty(winchance2, 1)



    for i in range(len(players)):
        print(i, players[i]);
