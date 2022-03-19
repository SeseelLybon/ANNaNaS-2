from __future__ import annotations

import uuid
import numpy as np
from typing import List
import math

import gym

from meeple import Meeple
from species import Species
from neuralnetwork import ConnectionHistory

import time

import maintools
log = maintools.colLogger("population")
from maintools import rng

class Population:

    def __init__(self, pop_size:int, input_size:int, output_size:int, isHollow=False ):
        self.UUID = uuid.uuid4();
        #self.ID = str(rng.bytes(4));
        self.innovationHistory:List[ConnectionHistory] = list()
        self.meeples:List[Meeple] = list()
        self.species:List[Species] = []
        self.nextSpeciesID = 0

        self.size = pop_size
        self.input_size = input_size
        self.output_size = output_size
        self.generation = 0

        self.maxStaleness = 100 # how often a species can not improve before it's considered stale/stuck

        self.genscoresHistor_max:List[float] = []#[0 for i in range(1000)];
        self.genscoresHistor_cur:List[float] = []#[0 for i in range(100)];
        self.scorehistogHistor:List[List[float]] = [[0 for i in range(20)] for i in range(100)];
        self.speciesScoreHistogram:List[List[float]] = list()

        for i in range(self.size):
            self.meeples.append(Meeple(input_size, output_size))
            self.meeples[-1].brain.mutate(self.innovationHistory)
            self.meeples[-1].brain.generateNetwork()

        self.bestMeeple:Meeple = self.meeples[0]
        self.highestFitness = 0
        self.highestScore = 0



    #update all the meeps that are currently alive
    def updateAlive(self):
        for meep in self.meeples:
            if meep.isAlive:
                #meep.look()
                #meep.think()
                #meep.update()
                pass;


    #returns bool if all the players are dead or done
    def isDone(self)-> bool:
        for meep in self.meeples:
            #if meep.isAlive or not meep.isDone: # Doesn't work atm
            if meep.isAlive:
                return False
        return True

    def countAlive(self)->int:
        tot = 0
        for meep in self.meeples:
            if meep.isAlive:
                tot+=1
        return tot


    def setBestMeeple(self):
        if self.bestMeeple:
            maxFit = self.highestFitness
        else:
            maxFit = 0

        #go through all meeples in the population and test if their fitness is higher than the previous one
        for meepi in range(self.size):
            if self.meeples[meepi].fitness > maxFit:
                self.bestMeeple = self.meeples[meepi]
                self.highestFitness = self.meeples[meepi].fitness
                self.highestScore = self.meeples[meepi].score
                maxFit = self.meeples[meepi].fitness



    def naturalSelection(self):
        for meep in self.meeples:
            if meep.score <=1:
                meep.score = 1;
        
        self.updateStats()
        self.print_deathrate()

        log.logger.info("Starting Natural Selection")

        log.logger.info("Speciating")
        species_pre_speciate = len(self.species)
        self.speciate()  # seperate the existing population into species for the purpose of natural selection

        id_s = []
        for spec in self.species:
            # Specie's ID
            # Amount of meeps in Specie
            # How stale Specie is
            # Highest fitness in Specie
            # Average fitness of Specie
            id_s.append((spec.speciesID, len(spec.meeples), spec.staleness ,round(spec.bestFitness,2), round(spec.averageFitness, 2)))
        id_s.sort(key=lambda x: x[3]); id_s.reverse();  id_s[:] = id_s[:5];

        log.logger.info("Species %d %d %d %s" % ( self.size,
                                           sum([len(x.meeples) for x in self.species]),
                                           len(self.species),
                                           id_s ) )

        species_pre_cull = len(self.species)
        log.logger.info("Sorting Species")
        self.calculateFitness()  # calc fitness of each meeple
        self.sortSpecies()  # sort all the species to the average fitness,best first. In the species sort by meeple's fitness

        # Clean the species
        log.logger.info("Culling Species")
        self.cullSpecies()
        self.setBestMeeple()

        log.logger.info("Killing Species")
        self.killStaleSpecies()
        self.killBadSpecies()

        log.logger.info("highest score %.4f" % self.highestScore)
        log.logger.info("highest fitness %.4f" % self.highestFitness)

        log.logger.info("Species %d:%d:%d" % (species_pre_speciate, species_pre_cull, len(self.species)))

        self.bestMeeple = self.bestMeeple.clone()
        #self.bestMeeple.sprite.color = (0,200,100)
        children:List[Meeple] = [self.bestMeeple]
        log.logger.info("Making new meeps from parents")

        averageSum = self.getAverageFitnessSum()
        if averageSum > 1:
            for specie in self.species:
                #add the best meeple of a specie to the new generation list
                children.append(specie.bestMeeple.clone())

                #generate number of children based on how well the species is doing compared to the rest; the better the bigger.
                newChildrenAmount = math.floor((specie.averageFitness/averageSum) * self.size - 1);

                for i in range(newChildrenAmount):
                    children.append(specie.generateChild(self.innovationHistory))
        else:
            log.logger.error("averageSum is 0, couldn't create a single meep from species");

        log.logger.info("Made %d new children from parents" % (len(children)-1))
        log.logger.info("Making new meeps from scratch")

        oldchillen = len(children)
        # If the pop-cap hasn't been filled yet, keep getting children from the best specie till it is filled
        #while len(children) < self.size:
        for dummy in range(self.size-len(children)):
            children.append( self.species[0].generateChild(self.innovationHistory) )

        log.logger.info("Made %d new children from scratch(best species)" % (len(children)-oldchillen))

        self.meeples[:] = children[:]
        self.generation += 1

        self.bestMeeple.brain.JSONStoreNeuralNetwork(filepath="BestMeepleBrain.json")
        self.population_dump();

        for meep in self.meeples:
            meep.brain.generateNetwork()



    def speciate(self):
        #clear meeps from existing species and reassign self.pop to self.species
        temp = len(self.species)
        for specie in self.species:
            specie.meeples.clear()

        for meep in self.meeples:
            speciesfound = False
            for specie in self.species:
                if specie.checkSameSpecies(meep, specie.bestMeeple):
                    specie.addToSpecies(meep)
                    speciesfound = True
                    break

            if not speciesfound:
                self.species.append(Species(meep=meep, speciesID=self.nextSpeciesID))
                self.nextSpeciesID+=1

        log.logger.info("Added %d new species" % (len(self.species)-temp))


    def calculateFitness(self):
        for meep in self.meeples:
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
            log.logger.warning("Killing %d stale species" % (prekill-len(self.species)))


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
        self.species[:] = self.species[0:2] + [specie for specie in self.species[2:] if averageSum > 0 and ((specie.averageFitness/averageSum) * self.size >= 1)]

        if prekill-len(self.species) > 0:
            log.logger.warning("Killing %d bad species" % (prekill-len(self.species)))


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


        self.genscoresHistor_cur.append(max(meep.score for meep in self.meeples[:]));
        self.genscoresHistor_cur[:] = self.genscoresHistor_cur[-100:]

        if len(self.genscoresHistor_max)==0:
            self.genscoresHistor_max.append(self.genscoresHistor_cur[0]);
        else:
            if self.genscoresHistor_max[-1] < self.genscoresHistor_cur[-1]:
                self.genscoresHistor_max.append(self.genscoresHistor_cur[-1]);
            else:
                self.genscoresHistor_max.append(self.genscoresHistor_max[-1]);

        self.genscoresHistor_max[:] = self.genscoresHistor_max[-1000:]



        bins = 20
        scorebin_cur = [0 for i in range(bins)];
        maxscore = self.genscoresHistor_cur[-1]
        step = maxscore/bins

        for meep in self.meeples:
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

        log.logger.info("scorebin bin:amount %s" % self.scorehistogHistor[-1]);


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


    def population_dump(self)->None:
        # Note to self; not using serpent because of cirular dependancy. No idea how to solve it.

        import jsonpickle
        import jsonpickle.ext.numpy as jsonpickle_numpy

        jsonpickle_numpy.register_handlers()
        jsonpickle.set_encoder_options('json', indent=4)
        #filepath = str(self.UUID)+".txt";
        #filepath = "dumps/"+str(self.generation)+"_Population.json";
        filepath = "dumps/"+"Population_"+str(self.generation%10)+".json";

        with open(filepath, 'w') as file:
            frozen = jsonpickle.encode(self)
            file.write(frozen);

    @staticmethod
    def population_load(filepath:str) -> Population:
        """

        Args:
            filepath: format: "dumps/+UUID+/generation_Population.json"

        Returns:

        """
        import jsonpickle;
        import jsonpickle.ext.numpy as jsonpickle_numpy;
        jsonpickle_numpy.register_handlers();
        jsonpickle.set_decoder_options('json');

        with open(filepath, 'r') as file:
            templines = file.readlines();

        tempjoined = ''.join(templines);

        thawed = jsonpickle.decode(tempjoined);

        return thawed

def deltaTimeS(last_time):
    return int((time.time()-last_time)//60)



import logging
import unittest
class TestPopulation(unittest.TestCase):
    def setUp(self)->None:
        log.logger.setLevel(logging.DEBUG)

    def tearDown(self)->None:
        pass;

    def test_populationDump(self)->None:
        testpop = Population(100,9,9);

        with self.subTest("Population Dump"):
            testpop.population_dump();
            pass;

        with self.subTest("Population Load"):
            testpop2 = Population.population_load("dumps/0_"+"Population.txt")
            pass;


    #@unittest.expectedFailure
    #def functioniexpecttofail(self):
    #   pass;

    #def someTest(self):
    #    with self.subTest("example test"):
    #        self.assertTrue(1==1);

if __name__ == "__main__":

    print("Starting population.py as main")

    #import timeit
    #import cProfile
    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()
    print("Finished population.py as main")
