from __future__ import annotations

from typing import List

import numpy as np
from numpy.random import default_rng

from node import Connection
from node import Node

rng = default_rng()

# NeuralNetwork = Genome


nextConnectionID:int = 10


class NeuralNetwork:

    def __init__(self, input_size:int, output_size:int, hollow:bool=False):
        self.input_size:int = input_size
        self.layers_total:int = 2
        self.output_size:int = output_size
        self.nextNodeID:int = 0
        self.nextConnectionID:int = 0
        self.biasNodeID:int

        self.connections:List[Connection] = list()
        self.nodes:List[Node] = list()
        self.network:List[Node] = list()

        self.mutateChance_AddNode = 0.03
        self.mutateChance_AddConnection = 0.30
        self.mutateChance_ChangeWeight = 0.75


        if not hollow:
            for nodei in range(self.input_size):
                self.nodes.append(Node(nodei))
                self.nodes[nodei].layer = 0
                self.nextNodeID += 1

            for nodei in range(self.input_size,self.input_size+self.output_size):
                self.nodes.append(Node(nodei))
                self.nodes[nodei].layer = 1
                self.nextNodeID += 1

            self.nodes.append(Node(self.nextNodeID))
            self.biasNodeID = self.nextNodeID
            self.nextNodeID+=1
            self.nodes[self.biasNodeID].layer = 0


    def getNode(self, nodeNumber:int):
        for nodei in range(len(self.nodes)):
            if self.nodes[nodei].ID == nodeNumber:
                return self.nodes[nodei]
        return None

    def connectNodes(self) -> None:
        for nodei in range(len(self.nodes)):
            self.nodes[nodei].outputConnections.clear()

        for conni in range(len(self.connections)):
            self.connections[conni].fromNode.outputConnections.append(self.connections[conni])

    def feedForward(self, inputValues:List[float]) -> List[float]:
        for nodei in range(self.input_size):
            self.nodes[nodei].outputValue=inputValues[nodei]
        self.nodes[self.biasNodeID].outputValue = 1

        for nodei in range(len(self.network)):
            self.network[nodei].fire()

        outputValues:List[float] = list()
        for outi in range(self.input_size, self.input_size+self.output_size):
            outputValues.append(self.nodes[outi].outputValue)

        for nodei in range(len(self.nodes)):
            self.nodes[nodei].inputSum = 0

        return outputValues

    def generateNetwork(self) -> None:
        self.connectNodes()
        self.network = list()

        for layeri in range(self.layers_total):
            for nodei in range(len(self.nodes)):
                if self.nodes[nodei].layer == layeri:
                    self.network.append(self.nodes[nodei])

    def addNode(self, innovationHistory:List[ConnectionHistory]) -> None:
        # if there's no connections, no point adding a new node yet
        if len(self.connections) == 0:
            self.addConnection(innovationHistory)
            return

        # pick a random Connection that isn't with the Bias node, and disable it.
        randomConnection:int = int(rng.integers(0, len(self.connections)))
        while( self.connections[randomConnection].fromNode == self.nodes[self.biasNodeID] ) and (len(self.connections) != 1):
            randomConnection = int(rng.integers(0, len(self.connections)))
        self.connections[randomConnection].enabled = False

        # create the new node
        newNodeNumber:int = self.nextNodeID
        self.nodes.append(Node(newNodeNumber))
        self.nextNodeID+=1

        # add new connection with weight 1 from the old connections fromNode to newNode
        connectionInnovationNumber:int = self.getInnovationNumber(innovationHistory,
                                                                  self.connections[randomConnection].fromNode,
                                                                  self.getNode(newNodeNumber))
        self.connections.append(Connection(self.connections[randomConnection].fromNode,
                                           self.getNode(newNodeNumber),
                                           1,
                                           connectionInnovationNumber))

        # add a new connection to the new node with the weight of the disabled connection
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory,
                                                              self.getNode(newNodeNumber),
                                                              self.connections[randomConnection].toNode)
        self.connections.append(Connection(self.getNode(newNodeNumber),
                                           self.connections[randomConnection].toNode,
                                           self.connections[randomConnection].weight,
                                           connectionInnovationNumber))
        self.getNode(newNodeNumber).layer = self.connections[randomConnection].fromNode.layer + 1

        # add new connection to the Bias node
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.getNode(self.biasNodeID), self.getNode(newNodeNumber))
        self.connections.append(Connection(self.getNode(self.biasNodeID), self.getNode(newNodeNumber) ,0 ,connectionInnovationNumber ) )

        # add a new layer if needed
        if self.getNode(newNodeNumber).layer == self.connections[randomConnection].toNode.layer:
            for nodei in range(len(self.nodes)-1):
                if self.nodes[nodei].layer >= self.getNode(newNodeNumber).layer:
                    self.nodes[nodei].layer+=1
            self.layers_total+=1

        self.connectNodes()





    def addConnection(self, innovationHistory:List[ConnectionHistory]) -> None:
        if self.isFullyConnected():
            print("addConnection failed, can't add new connection to filled network")
            return

        randomNode1:int = int(np.floor(rng.integers(0, len(self.nodes))))
        randomNode2:int = int(np.floor(rng.integers(0, len(self.nodes))))

        while self.checkValidityConnection(randomNode1, randomNode2):
            randomNode1 = int(np.floor(rng.integers(0, len(self.nodes))))
            randomNode2 = int(np.floor(rng.integers(0, len(self.nodes))))

        if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
            temp:int = randomNode2
            randomNode2 = randomNode1
            randomNode1 = temp

        connectionInnovationNumber:int = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        self.connections.append( Connection(self.nodes[randomNode1], self.nodes[randomNode2], rng.uniform(-1,1), connectionInnovationNumber) )
        self.connectNodes()


    def checkValidityConnection(self, r1:int, r2:int) -> bool:
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True
        return False

    def getInnovationNumber(self, innovationHistory:List[ConnectionHistory], fromNode:Node, toNode:Node)->int:
        global nextConnectionID
        isNew:bool = True
        connectionInnovationNumber:int = nextConnectionID
        for innoi in range(len(innovationHistory)):
            if innovationHistory[innoi].matches(self, fromNode, toNode):
                isNew = False # Not a new mutation
                connectionInnovationNumber = innovationHistory[innoi].innovationNumber
                break

        if isNew:
            innoNumbers:List[int] = list()
            for conni in range(len(self.connections)):
                innoNumbers.append(self.connections[conni].innovationNumber)

            innovationHistory.append(ConnectionHistory(fromNode.ID, toNode.ID, connectionInnovationNumber, innoNumbers))
            nextConnectionID+=1

        return connectionInnovationNumber

    def isFullyConnected(self)-> bool:
        maxConnections:int = 0
        nodesInLayers:np.ndarray = np.zeros([self.layers_total], int)

        for nodei in range(len(self.nodes)):
            nodesInLayers[self.nodes[nodei].layer] += 1

        for layeri in range(self.layers_total-1):
            nodesInFront:int = 0
            for layerj in range(layeri+1, self.layers_total):
                nodesInFront += nodesInLayers[layerj]

            maxConnections+= nodesInLayers[layeri]*nodesInFront

        if maxConnections == len(self.connections):
            return True
        else:
            return False

    def mutate(self, innovationHistory:List[ConnectionHistory])->None:
        if len(self.connections) == 0:
            self.addConnection(innovationHistory)

        rand1:float = rng.uniform()
        if rand1 < self.mutateChance_ChangeWeight:
            for conni in range(len(self.connections)):
                self.connections[conni].mutateWeight()

        rand2:float = rng.uniform()
        if rand2 < self.mutateChance_AddConnection:
            self.addConnection(innovationHistory)

        rand3: float = rng.uniform()
        if rand3 < self.mutateChance_AddNode:
            self.addNode(innovationHistory)

    def crossover(self, parent2:NeuralNetwork) -> NeuralNetwork:
        child:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size, hollow=True)
        child.connections.clear()
        child.nodes.clear()
        child.layers_total = self.layers_total
        child.nextNodeID = self.nextNodeID
        child.biasNodeID = self.biasNodeID

        childConnections:List[Connection] = list()
        isEnabled:List[bool] = list()

        for conni in range(len(self.connections)):
            setEnabled:bool = True
            parent2connection:int = self.matchingConnection(parent2, self.connections[conni].innovationNumber)
            if parent2connection != -1:
                if (not self.connections[conni].enabled) or (not parent2.connections[parent2connection].enabled):
                    if rng.uniform() < 0.75:
                        setEnabled= False
                if rng.uniform() <0.5:
                    childConnections.append(self.connections[conni])
                else:
                    childConnections.append(parent2.connections[parent2connection])
            else: # if they are the same
                childConnections.append(self.connections[conni])
                setEnabled = self.connections[conni].enabled
            isEnabled.append(setEnabled)

        for nodei in range(len(self.nodes)):
            child.nodes.append((self.nodes[nodei].clone()))

        for conni in range(len(childConnections)):
            child.connections.append(childConnections[conni].clone(child.getNode(childConnections[conni].fromNode.ID),
                                                                   child.getNode(childConnections[conni].toNode.ID)))
            child.connections[conni].enabled = isEnabled[conni]

        child.connectNodes()
        return child

    def matchingConnection(self, parent2:NeuralNetwork, innovationNumber:int)->int:
        for conni in range(len(parent2.connections)):
            if parent2.connections[conni].innovationNumber == innovationNumber:
                return conni
        return -1

    def clone(self) -> NeuralNetwork:
        clone:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size, hollow=True)
        for nodei in range(len(self.nodes)):
            clone.nodes.append(self.nodes[nodei].clone())

        for conni in range(len(self.connections)):
            clone.connections.append(self.connections[conni].clone(clone.getNode(self.connections[conni].fromNode.ID),
                                                                   clone.getNode(self.connections[conni].toNode.ID)))

        clone.layers_total = self.layers_total
        clone.nextNodeID = self.nextNodeID
        clone.biasNodeID = self.biasNodeID
        clone.connectNodes()

        return clone

    def printNetwork(self)->None:
        print("Print NN layers:", self.layers_total)
        print("Bias node: ", self.biasNodeID)
        print("Nodes:")
        for nodei in range(len(self.nodes)):
            print( self.nodes[nodei].ID, ",", end="" )
        print("\nConnections:", len(self.connections))
        for conni in range(len(self.connections)):
            print("Connection ", self.connections[conni].innovationNumber,
                  "\n\tfrom Node", self.connections[conni].fromNode.ID,
                  "\n\tto Node", self.connections[conni].toNode.ID,
                  "\n\tisEnabled", self.connections[conni].enabled,
                  "\n\tfrom layer", self.connections[conni].fromNode.layer,
                  "\n\tto layer", self.connections[conni].toNode.layer,
                  "\n\twith weight:", self.connections[conni].weight)




#   def get_output(self, num:int):
#   def get_outputs(self):
#   def set_input(self, num:int, intense:float):
#   def set_input(self, inputarray:np.ndarray):

#   def mutate(self, mutatechance=1/30, mutatestrength=1):
#   def clone(self):
#   def crossover(self, parent2): #parent2 = mate
#   def cost function

#   def train(self, training_data, training_answers, learnrate):
#   def GradientDescentDelta(self, training_input, desired_output, deltaimages=None):
#   def ApplyGradientDecentDelta(self, learnrate, DeltaOutputWeights, DeltaHiddenLayersWeights=None, batchsize=1):

#   def serpent_serialize(self):
#   def serpent_deserialize(self, pickledbrain):

#   def getSumWeights(self)->int:


#Here because of circular dependancy
class ConnectionHistory:
    def __init__(self, fromNode:int, toNode:int, innovationNumber:int, innovationNumbers:List[int]):
        self.fromNode:int = fromNode
        self.toNode:int = toNode
        self.innovationNumber:int = innovationNumber
        self.innovationNumbers:List[int] = innovationNumbers.copy()

    def matches(self, neuralnetwork:NeuralNetwork, fromNode:Node, toNode:Node) -> bool:
        if len(neuralnetwork.connections) == len(self.innovationNumbers):
            if fromNode.ID == self.fromNode and toNode.ID == self.toNode:
                for coni in range(len(neuralnetwork.connections)):
                    if not self.innovationNumbers.__contains__ (neuralnetwork.connections[coni].innovationNumber):
                        return False
                return True
        return False


if __name__ == "__main__":

    print("Starting neuralnetwork.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()

    '''
    doonce = False
    if doonce:
        innovationHistory:List[ConnectionHistory] = list()


        ANN1 = NeuralNetwork(7, 3)
        ANN1.generateNetwork()
        print("Made ANN1")
        output = ANN1.feedForward([1,2,3,4,5,6,7])
        print("ANN1 feedForward", output)
        ANN1.printNetwork()
        print("Mutating")
        for dummy in range(10):
            ANN1.mutate(innovationHistory)
        ANN1.printNetwork()
        ANN1.generateNetwork()

        output = ANN1.feedForward([1,2,3,4,5,6,7])
        print(output)
        print("ANN1 Mutated feedForward",output)

        print("Mutating")
        for dummy in range(10):
            ANN1.mutate(innovationHistory)
        ANN1.printNetwork()
        ANN1.generateNetwork()

        output = ANN1.feedForward([1,2,3,4,5,6,7])
        print(output)
        print("ANN1 Mutated feedForward",output)
    '''

    print("Finished neuralnetwork.py as main")