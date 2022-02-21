from __future__ import annotations


# https://en.wikipedia.org/wiki/Elo_rating_system



class ELO:
    Kfactor = 32

def rate_1v1(winner:Rating, loser:Rating, score:float, isDraw=False):
    newRating(winner, winChance(winner, loser), score)
    newRating(loser, winChance(loser, winner), 1-score)
    return;

# Probability of playerA winning over PlayerB
def winChance(Ra:Rating, Rb:Rating):
    step1 = Rb.rating-Ra.rating
    step2 = step1/400
    step3 = 10**step2
    step4 = 1+step3
    return 1/step4

def newRating(rating:Rating, winchance:float, win:float):
    rating.rating = rating.rating+ELO.Kfactor* (win - winchance)

class Rating:
    def __init__(self, rating:int=1000):
        self.wins = 0;
        self.losses = 0;
        self.draws = 0;

        self.uncertainty = 1;

        self.rating = rating
        return;

    '''Uncertainty
    Prototype variable for later
    Uncertainty indicates how uncertain I am that rating is the correct score.
    Lowers impact of K-factor.
    Performing as expected lowers it, unexpected results increase it.
    So a player with 1500 and uncertainty 1 is new,
    where as a 1500 player with uncerainty 0.1 has been playing for a while and just happens to rest at the entrance ELO
    '''


if __name__ == "__main__":
    player1:Rating = Rating(1000)
    player2:Rating = Rating(1000)

    winchance1 = winChance(player1, player2)
    winchance2 = winChance(player2, player1)

    print(winchance1, 1-winchance2)
    print(winchance2, 1-winchance1)

    newRating(player1, winChance(player1, player2), 0)
    newRating(player2, winChance(player2, player1), 1)

    print(player1)
    print(player2)