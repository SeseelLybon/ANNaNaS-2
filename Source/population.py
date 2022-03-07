from __future__ import annotations

import numpy as np
from typing import List
import math

from meeple import Meeple
from species import Species
from neuralnetwork import NeuralNetwork
from neuralnetwork import ConnectionHistory

import time

import colorlog
import structlog
import logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
logger = colorlog.getLogger('population')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Population:

    def __init__(self, pop_size:int, input_size:int, output_size:int ):
        self.innovationHistory:List[ConnectionHistory] = list()
        self.pop:List[Meeple] = list()
        self.species:List[Species] = []
        self.nextSpeciesID = 0

        self.size = pop_size
        self.input_size = input_size
        self.output_size = output_size
        self.generation = 0

        self.maxStaleness = 50 # how often a species can not improve before it's considered stale/stuck
        self.massExtinctionEvent = False

        self.genscoresHistor_max:List[float] = [0 for i in range(1000)];
        self.genscoresHistor_cur:List[float] = [0 for i in range(100)];
        self.scorehistogHistor:List[List[float]] = [[0 for i in range(20)] for i in range(100)];
        self.speciesScoreHistogram:List[List[float]] = list()

        for i in range(self.size):
            self.pop.append( Meeple(input_size, output_size) )
            self.pop[-1].brain.mutate(self.innovationHistory)
            self.pop[-1].brain.generateNetwork()

        self.bestMeeple:Meeple = self.pop[0]
        self.highestFitness = 0
        self.highestScore = 0



    #update all the meeps that are currently alive
    def updateAlive(self):
        for meep in self.pop:
            if meep.isAlive:
                meep.look()
                meep.think()
                meep.update()


    #returns bool if all the players are dead or done
    def isDone(self)-> bool:
        for meep in self.pop:
            #if meep.isAlive or not meep.isDone: # Doesn't work atm
            if meep.isAlive:
                return False
        return True

    def countAlive(self)->int:
        tot = 0
        for meep in self.pop:
            if meep.isAlive:
                tot+=1
        return tot


    def setBestMeeple(self):
        if self.bestMeeple:
            maxFit = self.highestFitness
        else:
            maxFit = 0

        #go through all meeples in the population and test if their fitness is higher than the previous one
        for meepi in range(len(self.pop)):
            if self.pop[meepi].fitness > maxFit:
                self.bestMeeple = self.pop[meepi]
                self.highestFitness = self.pop[meepi].fitness
                self.highestScore = self.pop[meepi].score
                maxFit = self.pop[meepi].fitness

        self.bestMeeple.brain.JSONstoreNeuralNetwork(filepath="BestMeepleBrain.json")


    def naturalSelection(self):

        self.updateStats()
        self.print_deathrate()

        logger.info("Starting Natural Selection")

        logger.info("Speciating")
        species_pre_speciate = len(self.species)
        self.speciate()  # seperate the existing population into species for the purpose of natural selection

        id_s = []
        for spec in self.species:
            # Specie's ID
            # Amount of meeps in Specie
            # How stale Specie is
            # Highest fitness in Specie
            # Average fitness of Specie
            id_s.append((spec.speciesID, len(spec.meeples), spec.staleness ,spec.bestFitness, spec.averageFitness))
        id_s.sort(key=lambda x: x[3]); id_s.reverse();  id_s[:] = id_s[:5];

        logger.info("Species %d %d %d %s" % ( self.size,
                                           sum([len(x.meeples) for x in self.species]),
                                           len(self.species),
                                           id_s ) )

        species_pre_cull = len(self.species)
        logger.info("Sorting Species")
        self.calculateFitness()  # calc fitness of each meeple
        self.sortSpecies()  # sort all the species to the average fitness,best first. In the species sort by meeple's fitness

        # Clean the species
        logger.info("Culling Species")
        self.cullSpecies()
        self.setBestMeeple()

        logger.info("Killing Species")
        self.killStaleSpecies()
        self.killBadSpecies()

        logger.info("highest score %.4f" % self.highestScore)
        logger.info("highest fitness %.4f" % self.highestFitness)

        logger.info("Species %d:%d:%d" % (species_pre_speciate, species_pre_cull, len(self.species)))

        self.bestMeeple = self.bestMeeple.clone()
        #self.bestMeeple.sprite.color = (0,200,100)
        children:List[Meeple] = [self.bestMeeple]
        logger.info("Making new meeps from parents")

        averageSum = self.getAverageFitnessSum()
        for specie in self.species:
            #add the best meeple of a specie to the new generation list
            children.append(specie.bestMeeple.clone())

            #generate number of children based on how well the species is doing compared to the rest; the better the bigger.
            newChildrenAmount = math.floor((specie.averageFitness/averageSum) * len(self.pop) ) -1

            for i in range(newChildrenAmount):
                children.append(specie.generateChild(self.innovationHistory))

        logger.info("Made %d new children from parents" % (len(children)-1))
        logger.info("Making new meeps from scratch")

        oldchillen = len(children)
        # If the pop-cap hasn't been filled yet, keep getting children from the best specie till it is filled
        #while len(children) < self.size:
        for dummy in range(self.size-len(children)):
            children.append( self.species[0].generateChild(self.innovationHistory) )

        logger.info("Made %d new children from scratch(best species)" % (len(children)-oldchillen))

        self.pop[:] = children[:]
        self.generation += 1

        for meep in self.pop:
            meep.brain.generateNetwork()



    def speciate(self):
        #clear meeps from existing species and reassign self.pop to self.species
        temp = len(self.species)
        for specie in self.species:
            specie.meeples.clear()

        for meep in self.pop:
            speciesfound = False
            for specie in self.species:
                if specie.checkSameSpecies(meep, specie.bestMeeple):
                    specie.addToSpecies(meep)
                    speciesfound = True
                    break

            if not speciesfound:
                self.species.append(Species(meep=meep, speciesID=self.nextSpeciesID))
                self.nextSpeciesID+=1

        logger.info("Added %d new species" % (len(self.species)-temp))


    def calculateFitness(self):
        for meep in self.pop:
            meep.calculateFitness()


    #sort the population of a species by fitness
    #sort the species by the average of the species
    def sortSpecies(self):
        for specie in self.species:
            specie.sortSpecies()


    def killStaleSpecies(self):

        prekill = len(self.species)

        # protect 2 fittest species from staleness, then add rest of not stale species.
        self.species[:] = self.species[0:2] + [ specie for specie in self.species[2:] if specie.staleness < self.maxStaleness]
        #self.species[:] = [ specie for specie in self.species if specie.staleness < self.maxStaleness]

        if prekill-len(self.species) > 0:
            logger.warning("Killing %d stale species" % (prekill-len(self.species)))


    def killBadSpecies(self):


        prekill = len(self.species)

        self.species[:] = [ specie for specie in self.species if len(specie.meeples) > 0 ]

        for specie in self.species:
            #specie.fitnessSharing()
            specie.fitnessSharing_alt()
            #specie.fitnessSharing_book()
            specie.setAverageFitness()

        averageSum = self.getAverageFitnessSum()

        #self.species[:] = [ specie for specie in self.species if ((specie.averageFitness/averageSum) * len(self.pop) >= 1) ]
        self.species[:] = self.species[0:2] + [ specie for specie in self.species[2:] if averageSum > 0 and ((specie.averageFitness/averageSum) * len(self.pop) >= 1) ]

        if prekill-len(self.species) > 0:
            logger.warning("Killing %d bad species" % (prekill-len(self.species)))


    #get the sum of averages from each specie
    def getAverageFitnessSum(self)->float:
        tempsum = 0
        for specie in self.species:
            tempsum += specie.averageFitness
        return tempsum

    def cullSpecies(self):
        # remove the bottom half of all species.
        for specie in self.species:
            specie.cull()


    def updateStats(self):


        self.genscoresHistor_cur.append(max(meep.score for meep in self.pop[:]));
        self.genscoresHistor_cur[:] = self.genscoresHistor_cur[-100:]

        if self.genscoresHistor_max[-1] < self.genscoresHistor_cur[-1]:
            self.genscoresHistor_max.append(self.genscoresHistor_cur[-1]);
        else:
            self.genscoresHistor_max.append(self.genscoresHistor_max[-1]);

        self.genscoresHistor_max[:] = self.genscoresHistor_max[-1000:]



        bins = 20
        scorebin_cur = [0 for i in range(bins)];
        maxscore = self.genscoresHistor_cur[-1]
        step = maxscore/bins

        for meep in self.pop:
            for i in range(0,bins):
                if step*i <= meep.score <= step*(i+1):
                    scorebin_cur[i]+=1;
                    break;
            continue;

        self.scorehistogHistor.append(scorebin_cur)
        self.scorehistogHistor[:] = self.scorehistogHistor[-100:]


        bins = 20
        self.speciesScoreHistogram.clear()
        if len(self.species) > 0:
            for specie in self.species:
                speciesscorebin = [0 for i in range(bins)]
                maxscore = max([meep.score for meep in specie.meeples])
                step = maxscore/bins

                for meep in specie.meeples:
                    for i in range(0,bins):
                        if step*i <= meep.score <= step*(i+1):
                            speciesscorebin[i]+=1;
                            break;
                    continue;
                self.speciesScoreHistogram.append(speciesscorebin)
        else:
            self.speciesScoreHistogram.append([0 for i in range(bins)])

        logger.info("scorebin bin:amount %s" % self.scorehistogHistor[-1]);


    def print_deathrate(self, do_print=True):
        if not do_print:
            return

        with open("spreadsheetdata.txt", "a") as f:
            now = time.localtime();
            f.write( "%d-%d  \t%d:%d:%d\t%.2f\t%.2f\t%d\n"
                     % (now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min, now.tm_sec,
                        self.genscoresHistor_max[-1],
                        self.genscoresHistor_cur[-1],
                        self.generation))


def deltaTimeS(last_time):
    return int((time.time()-last_time)//60)















if __name__ == "__main__":

    print("Starting population.py as main")

    #import timeit
    #import cProfile
    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()
    print("Finished population.py as main")
