from __future__ import annotations

import math
from itertools import combinations
import numpy as np

from population import Population
import maintools
from maintools import rng
log = maintools.colLogger("binarytodecimal")
from meeple import Meeple

from typing import Tuple


# Use "from this import *" ?


def binarytodecimalMain(population:Population):
    test = np.array([[0,0,0],
                     [1,0,0],
                     [0,1,0],
                     [1,1,0],
                     [0,0,1],
                     [1,0,1],
                     [0,1,1],
                     [1,1,1]]);
    answer = np.linspace(0,test.shape[0]);
    for meep in population.meeples:
        for t, a in zip(test, answer):
            meep.think(vision=[1,0]+board)
            decision = meep.decision
            index = decision.index(max(decision))