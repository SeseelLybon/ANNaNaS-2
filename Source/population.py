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

        self.print_deathrate()
        UnMassExtingtionEventsAttempt = 0
        species_pre_speciate:int = -1
        species_pre_cull:int = -1

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
        id_s.sort(key=lambda x: x[3]); id_s.reverse();  id_s[:] = id_s[:50];

        logger.info("Species %d %d %d %s" % ( self.size,
                                           sum([len(x.meeples) for x in self.species]),
                                           len(self.species),
                                           id_s ) )

        species_pre_cull = len(self.species)
        logger.info("Sorting Species")
        self.calculateFitness()  # calc fitness of each meeple
        self.sortSpecies()  # sort all the species to the average fitness,best first. In the species sort by meeple's fitness

        if self.massExtinctionEvent:
            self.massExtingtion()
            self.massExtinctionEvent = False

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

    def massExtingtion(self) -> None:
        self.species=self.species[:5]

    def print_deathrate(self, do_print=True):
        if not do_print:
            return
        # go through all meeps and add their score to a dict.
        # pick the highest score and bins for every x% of score from the max
        # print
        scoredict = dict()
        for meep in self.pop:
            if meep.score in scoredict:
                scoredict[meep.score] += 1
            else:
                scoredict[meep.score] = 1

        highestscore = max(scoredict.keys())

        scorebins = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        for meep in self.pop:
            score = round( meep.score / max(highestscore*0.1, 1), 0)
            if score in scorebins:
                scorebins[score] += 1
            else:
                scorebins[score] = 1

        newline:str = ""
        with open("spreadsheetdata.txt", "a") as f:
            temp_string = ""
            for value in scorebins.values():
                temp_string+= "\t" + str(value)

            f.write(str(time.time()) + "\t" +
                    str(self.highestScore) + "\t" +
                    str(max(self.pop, key=lambda kv: kv.score).score) + "\t" +
                    str(self.generation) + "\n")# +
                    #temp_string + "\n")
            # Time, Highest score overall, highst score generation, generation, deathbin



            #for specie in self.species[:max(10, len(self.species))]:
            #    specie.sortSpecie()
            #    newline += "\t" + \
            #              str(specie.speciesID) + "\t" + \
            #              str(len(specie.meeples)) + "\t" + \
            #              str(specie.meeples[0].fitness) + "\t" + \
            #              str(specie.meeples[len(specie.meeples)//2] ) + "\t" + \
            #              str(specie.meeples[-1].fitness) + "\t" + \
            #              str(specie.averageFitness) + "\t" + "\n"
            #    f.write( newline )
        #for key, value in sorted(scorebins.items(), key=lambda kv: kv[0]):
        #    print(key,":",value, " - ")
        logger.info("scorebin bin:amount %s" % sorted(scorebins.items(), key=lambda kv: kv[0]))


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
