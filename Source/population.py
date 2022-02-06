from __future__ import annotations

import numpy as np
from typing import List
import math

from meeple import Meeple
from species import Species
from neuralnetwork import NeuralNetwork
from neuralnetwork import ConnectionHistory

import time



class Population:

    def __init__(self, pop_size:int, input_size:int, output_size:int ):
        self.innovationHistory:ConnectionHistory = ConnectionHistory()
        self.pop = np.ndarray([pop_size], dtype=Meeple)
        self.species:List[Species] = []
        self.speciesCreated = 0

        self.size = pop_size
        self.input_size = input_size
        self.output_size = output_size
        self.generation = 0

        self.training_data = training_data
        self.training_answers = training_answers

        self.maxStaleness = 15 # how often a species can not improve before it's considered stale/stuck
        self.massExtinctionEvent = False

        for i in range(self.pop.shape[0]):
            self.pop[i] = Meeple(input_size, output_size)
            self.pop[i].brain.generateNetwork()
            self.pop[i].brain.mutate(innovationHistory)

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

        maxFit = 0

        #go through all meeples in the population and test if their fitness is higher than the previous one
        for specie_i in range(len(self.species)):
            if self.species[specie_i].bestFitness > maxFit:
                maxFit = self.species[specie_i].bestFitness

                self.bestMeeple = self.species[specie_i].bestMeeple
                self.highestFitness = self.bestMeeple.brain.fitness
                self.highestScore = self.bestMeeple.brain.score


    def naturalSelection(self):

        last_time = time.time()

        self.print_deathrate()
        runonce = True
        UnMassExtingtionEventsAttempt = 0
        species_pre_speciate:int = -1
        species_pre_cull:int = -1

        print(deltaTimeS(last_time),"s - Starting Natural Selection")



        print(deltaTimeS(last_time), "s - Speciating")
        species_pre_speciate = len(self.species)
        self.speciate()  # seperate the existing population into species for the purpose of natural selection
        species_pre_cull = len(self.species)
        print(deltaTimeS(last_time), "s - Sorting Species")
        self.calculateFitness()  # calc fitness of each meeple, currently not needed
        self.sortSpecies()  # sort all the species to the average fitness, best first. In the species sort by meeple's fitness

        if self.massExtingtionEvent:
            self.massExtingtion()
            self.massExtingtionEvent = False

        # Clean the species
        print(deltaTimeS(last_time), "s - Culling Species")
        self.cullSpecies()
        self.setBestMeeple()

        print(deltaTimeS(last_time), "s - Killing Species")
        self.killBadSpecies()
        self.killStaleSpecies()




        print("highest score", self.highestScore)
        print("highest fitness", self.highestFitness)

        if species_pre_cull - species_pre_speciate > 0:
            print("Added", species_pre_cull - species_pre_speciate, "new species")

        print(deltaTimeS(last_time), "s- Species prespeciate:precull:postcull", species_pre_speciate, species_pre_cull, len(self.species))

        id_s = []
        for spec in self.species:
            # Specie's ID
            # Amount of meeps in Specie
            # How stale Specie is
            # Highest fitness in Specie
            # Average fitness of Specie
            id_s.append((spec.speciesID, len(spec.meeples),spec.staleness,spec.bestFitness, spec.averageFitness))
        id_s.sort(key=lambda x: x[4])
        id_s.reverse()
        id_s[:] = id_s[:50]
        print(deltaTimeS(last_time), "s- Species ID's", id_s )

        self.bestMeeple = self.bestMeeple.clone()
        #self.bestMeeple.sprite.color = (0,200,100)
        children:List[Meeple] = [self.bestMeeple]

        print(deltaTimeS(last_time), "s- Making new meeps from parents")

        # TODO: normalize the diffirence in fitness so that the number doesn't get stupidly big as easily.
        #   something something log(e**x).

        for specie in self.species:
            #add the best meeple of a specie to the new generation list
            children.append(specie.bestMeeple.clone())

            #generate number of children based on how well the species is doing compared to the rest; the better the bigger.
            newChildrenAmount = math.floor((specie.averageFitness/self.getAverageFitnessSum()) *self.pop.size) -1

            for i in range(newChildrenAmount):
                children.append(specie.generateChild())

        print(deltaTimeS(last_time), "s- Making new meeps from scratch")

        # If the pop-cap hasn't been filled yet, keep getting children from the best specie till it is filled
        while len(children) < self.size:
            children.append( self.species[0].generateChild() )

        self.pop = np.array(children, dtype=Meeple)
        self.generation += 1



    def speciate(self):
        #clear meeps from existing species and reassign self.pop to self.species
        for specie in self.species:
            specie.meeples.clear()

        for meep in self.pop:
            speciesfound = False
            for specie in self.species:
                if specie.checkSimilarSpecies(meep):
                    specie.addToSpecies(meep)
                    speciesfound = True
                    break
            if not speciesfound:
                self.species.append(Species(meep=meep, speciesID=self.speciesCreated))
                self.speciesCreated+=1


    def calculateFitness(self):
        for meep in self.pop:
            meep.calculateFitness()


    #sort the population of a species by fitness
    #sort the species by the average of the species
    def sortSpecies(self):
        for specie in self.species:
            specie.sortSpecie()

        self.species.sort(key=lambda specie: specie.averageFitness)

    def killStaleSpecies(self):

        markedForRemoval = list()

        for specie in self.species:
            if specie.staleness >= self.maxStaleness:
                markedForRemoval.append(specie)

        if len(markedForRemoval) > 0:
            print("Killing", len(markedForRemoval), "stale species")
        self.species[:] = [ x for x in self.species if x not in markedForRemoval ]


    def killBadSpecies(self):

        averageSum = getAvgFitnessSum()

        markedForRemoval = list()

        for specie in self.species:
            # this calculates how many children a specie is allowed to produce in Population.naturalSelection()
            # If this is less then one, the specie did so bad, it won't generate a child then. So it basically just died here.
            if (specie.averageFitness/averageSum) * len(self.pop) < 1:
                markedForRemoval.append(specie)

        if len(markedForRemoval) > 0:
            print("Killing", len(markedForRemoval), "bad species")

        self.species[:] = [ x for x in self.species if x not in markedForRemoval ]

    #get the sum of averages from each specie
    def getAverageFitnessSum(self)->float:
        tempsum = 0
        for specie in self.species:
            tempsum+= specie.averageFitness
        return tempsum

    def cullSpecies(self):
        # remove the bottom half of all species.
        for specie in self.species:
            specie.cull()
            specie.fitnessSharing()
            specie.generateAverageFitness()

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
            if meep.brain.score in scoredict:
                scoredict[meep.brain.score] += 1
            else:
                scoredict[meep.brain.score] = 1

        highestscore = max(scoredict.keys())

        scorebins = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        for meep in self.pop:
            score = round( meep.brain.score / max(highestscore*0.1, 1), 0)
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
                    str(max(self.pop, key=lambda kv: kv.brain.score).brain.score) + "\t" +
                    str(self.generation) + "\n")# +
                    #temp_string + "\n")
            # Time, Highest score overall, highst score generation, generation, deathbin



            #for specie in self.species[:max(10, len(self.species))]:
            #    specie.sortSpecie()
            #    newline += "\t" + \
            #              str(specie.speciesID) + "\t" + \
            #              str(len(specie.meeples)) + "\t" + \
            #              str(specie.meeples[0].brain.fitness) + "\t" + \
            #              str(specie.meeples[len(specie.meeples)//2] ) + "\t" + \
            #              str(specie.meeples[-1].brain.fitness) + "\t" + \
            #              str(specie.averageFitness) + "\t" + "\n"
            #    f.write( newline )
        #for key, value in sorted(scorebins.items(), key=lambda kv: kv[0]):
        #    print(key,":",value, " - ")
        print("death bin:amount,", sorted(scorebins.items(), key=lambda kv: kv[0]))
        print(newline)


def deltaTimeS(last_time):
    return int((time.time()-last_time)//60)















if __name__ == "__main__":

    import timeit
    import cProfile
    print("Starting population.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()
    print("Finished population.py as main")
