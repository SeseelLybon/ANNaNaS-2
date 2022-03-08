from __future__ import annotations


from typing import List

import math
import numpy as np
from itertools import combinations

from node import Connection
from node import Node
from maintools import rng

from pymunk import Vec2d

import pyglet
import pyglet.gl as pygl


# NeuralNetwork = Genome


nextConnectionID:int = 10
nextNeuralNetworkID:int = 10

import maintools
log = maintools.colLogger("neuralnetwork")

class NeuralNetwork:

    def __init__(self, input_size:int, output_size:int, ID:int=-1, hollow:bool=False):
        global nextNeuralNetworkID
        self.input_size:int = input_size
        self.layers_amount:int = 2
        self.output_size:int = output_size
        self.nextNodeID:int = 0
        self.biasNodeID:int
        if ID == -1:
            self.ID = nextNeuralNetworkID
            nextNeuralNetworkID+=1
        else:
            self.ID = ID

        self.connections:List[Connection] = list()
        self.nodes:List[Node] = list()
        self.network:List[Node] = list()

        if not hollow:
            for nodei in range(self.input_size):
                self.nodes.append(Node(nodei))
                self.nodes[nodei].layer = 0
                self.nextNodeID += 1

            for nodei in range(self.input_size,self.input_size+self.output_size):
                self.nodes.append(Node(nodei))
                self.nodes[nodei].layer = 1
                self.nextNodeID += 1

            # Connect all inputs to outputs
            #for nodei in range(self.input_size):
            #    self.addConnection()

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

    def feedForward(self, vision:List[float], postClean=True) -> List[float]:

        # Set input nodes with vision vector
        for nodei in range(self.input_size):
            self.nodes[nodei].outputValue=vision[nodei]
        self.nodes[self.biasNodeID].outputValue = 1


        for nodei in range(len(self.network)):
            self.network[nodei].fire(self.layers_amount)

        outputValues:List[float] = list()
        for outi in range(self.input_size, self.input_size+self.output_size):
            outputValues.append(self.nodes[outi].outputValue)

        for nodei in range(len(self.nodes)):
            self.nodes[nodei].inputSum = 0

        return outputValues

    def generateNetwork(self) -> None:
        self.connectNodes()
        self.network = list()

        for layeri in range(self.layers_amount):
            for nodei in range(len(self.nodes)):
                if self.nodes[nodei].layer == layeri:
                    self.network.append(self.nodes[nodei])

    def addNode(self, innovationHistory:List[ConnectionHistory]) -> None:
        # if there's no connections, no point adding a new node yet
        if len(self.connections) == 0:
            self.addConnection(innovationHistory)
            return

        log.logger.debug("Adding new Node")

        # pick a random Connection that isn't with the Bias node, and disable it.
        rnglist =  list(range(len(self.connections)))
        rng.shuffle(rnglist)
        randomConnection = None
        for rngcon in rnglist:
            if self.connections[rngcon].fromNode != self.nodes[self.biasNodeID]:
                self.connections[rngcon].enabled = False
                randomConnection = rngcon
                break

        # there's no available connection, might as well add one
        if randomConnection is None:
            self.addConnection(innovationHistory)
            return

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
            self.layers_amount+=1

        self.connectNodes()

        log.logger.debug("Added new Node, replacing %d:%d with %d:%d:%d" % (self.connections[randomConnection].fromNode.ID,
                                                                        self.connections[randomConnection].toNode.ID,
                                                                        self.connections[randomConnection].fromNode.ID,
                                                                        newNodeNumber,
                                                                        self.connections[randomConnection].toNode.ID))





    def addConnection(self, innovationHistory:List[ConnectionHistory]) -> None:
        if self.isFullyConnected():
            #print("addConnection failed, can't add new connection to filled network")
            return
        log.logger.debug("Adding new Connection")

        # grab 2 nodes that don't have a connection
        rnglist = list( combinations(list(range(len(self.nodes))), 2))
        rng.shuffle(rnglist)
        randomNode1:int = None
        randomNode2:int = None
        for rngconi1, rngconi2 in rnglist:
            randomNode1 = rngconi1
            randomNode2 = rngconi2
            if randomNode1 != self.nodes[self.biasNodeID] and \
                    randomNode2 != self.nodes[self.biasNodeID]:
                if not self.checkIfConnected(randomNode1, randomNode2):
                    break;

        log.logger.debug("%s, %s : these meeps were considered non-duplicate" % (randomNode1, randomNode2))

        if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
            temp:int = randomNode2
            randomNode2 = randomNode1
            randomNode1 = temp

        connectionInnovationNumber:int = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        self.connections.append( Connection(self.nodes[randomNode1], self.nodes[randomNode2], rng.uniform(-1,1), connectionInnovationNumber) )
        self.connectNodes()


        log.logger.debug("Added new Connection: %d:%d" % (randomNode1, randomNode2))



    def checkIfConnected(self, r1:int, r2:int) -> bool:
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True
        if self.nodes[r2].isConnectedTo(self.nodes[r1]):
            return True
        return False

    def getInnovationNumber(self, innovationHistory:List[ConnectionHistory], fromNode:Node, toNode:Node)->int:
        global nextConnectionID
        isNew:bool = True
        connectionInnovationNumber:int = nextConnectionID

        # Check if tried innovation already exists, then use that innovation
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
        nodesInLayers:np.ndarray = np.zeros([self.layers_amount], int)

        for nodei in range(len(self.nodes)):
            nodesInLayers[self.nodes[nodei].layer] += 1

        for layeri in range(self.layers_amount - 1):
            nodesInFront:int = 0
            for layerj in range(layeri+1, self.layers_amount):
                nodesInFront += nodesInLayers[layerj]

            maxConnections+= nodesInLayers[layeri]*nodesInFront

        if maxConnections == len(self.connections):
            return True
        else:
            return False

    def mutate(self, innovationHistory:List[ConnectionHistory], staleness=0)->None:
        log.logger.debug("Mutating")
        if len(self.connections) == 0:
            log.logger.debug("No connections to mutate, adding new connection")
            self.addConnection(innovationHistory)

        roll = rng.uniform()

        # Random thing that makes the population mutate the staler it gets.
        if staleness == 0:
            stalenessMod:float = 1
        else:
            stalenessMod:float = 1+staleness/500

        if rng.uniform() < 0.01*stalenessMod:
            log.logger.debug("RNG: Add new node")
            prenoduples = getDuplicateConnections(self)
            self.addNode(innovationHistory)
            if len(getDuplicateConnections(self)) != len(prenoduples):
                log.logger.fatal("Adding Node caused duplicate")
                printDuplicateConnections(self)

        elif rng.uniform() < 0.05*stalenessMod:
            log.logger.debug("RNG: Add new connection")
            prenoduples = getDuplicateConnections(self)
            self.addConnection(innovationHistory)
            if len(getDuplicateConnections(self)) != len(prenoduples):
                log.logger.fatal("Adding Connection caused duplicate")
                printDuplicateConnections(self)

        elif rng.uniform() < 0.80*stalenessMod:
            log.logger.debug("RNG: Mutate weights")
            for conni in range(len(self.connections)):
                self.connections[conni].mutateWeight()

    def crossover(self, parent2:NeuralNetwork) -> NeuralNetwork:
        global nextNeuralNetworkID
        child:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size, ID=nextNeuralNetworkID, hollow=True)
        child.connections.clear()
        child.nodes.clear()
        child.layers_amount = self.layers_amount
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
        clone:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size, self.ID, hollow=True)
        for nodei in range(len(self.nodes)):
            clone.nodes.append(self.nodes[nodei].clone())

        for conni in range(len(self.connections)):
            clone.connections.append(self.connections[conni].clone(clone.getNode(self.connections[conni].fromNode.ID),
                                                                   clone.getNode(self.connections[conni].toNode.ID)))
        clone.layers_amount = self.layers_amount
        clone.nextNodeID = self.nextNodeID
        clone.biasNodeID = self.biasNodeID
        clone.connectNodes()

        return clone

    def printNetwork(self)->None:
        print("Print NN layers:", self.layers_amount)
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



    def drawNetwork(self, startX:int, startY:int, width:int, height:int)->None:
        # batchConnections = pyglet.graphics.Batch()
        # batchNodesOutlines = pyglet.graphics.Batch()
        # batchNodes = pyglet.graphics.Batch()
        # batchLabels = pyglet.graphics.Batch()

        # batch.add(x)
        # batch.draw()

        allNodes:List[List[Node]] = []
        nodePoses:List[Vec2d] = []
        nodeNumbers:List[int] = []

        # gather all nodes in what is essentially self.network
        for layeri in range(self.layers_amount):
            temp:List[Node] = []
            for nodei in range(len(self.nodes)):
                if self.nodes[nodei].layer == layeri:
                    temp.append(self.nodes[nodei])
            allNodes.append(temp)

        # setup all the node positions
        for layeri in range(self.layers_amount):
            x:int = int(startX+((layeri)*width)/
                        (self.layers_amount + 1.0))
            for nodei in range(len(allNodes[layeri])):

                #if layeri%2==1 and not layeri == self.layers_amount:
                #    y:int = int(startY + ((nodei*height)/(len(allNodes[layeri])+1)+
                #                          (height/2)/(len(allNodes[layeri])+1)))
                #else:
                #    y:int = int(startY + ((nodei*height)/(len(allNodes[layeri])+1))
                y:int = int(startY + ((nodei*height)/(len(allNodes[layeri])+1)))

                nodePoses.append(Vec2d(x, y))
                nodeNumbers.append(allNodes[layeri][nodei].ID)

        # draw all the connections
        for conni in range(len(self.connections)):
            if self.connections[conni].enabled:
                pygl.glLineWidth(np.abs(int(self.connections[conni].weight*2))+1)
            else:
                continue

            if self.connections[conni].weight >=0:
                col = (255, 0, 0, #red/positive weight
                       255, 0, 0)
            else:
                col = (0, 0, 255, #blue/negative weight
                       0, 0, 255)

            fromNode_pos:Vec2d = nodePoses[ nodeNumbers.index( self.connections[conni].fromNode.ID ) ]
            toNode_pos:np.Vec2d = nodePoses[ nodeNumbers.index( self.connections[conni].toNode.ID ) ]

            pyglet.graphics.draw(2, pygl.GL_LINES, ('v2i', (fromNode_pos.x,
                                                            fromNode_pos.y,
                                                            toNode_pos.x,
                                                            toNode_pos.y) ),
                                                            ('c3B', col))
        # Draw all nodes (and ID's)

        label = pyglet.text.Label('23423423',
                                  font_name='Times New Roman',
                                  font_size=14,
                                  x=100, y=100,
                                  anchor_x='center', anchor_y='center',
                                  color=(0,0,0, 255))
        nodeShapeOutline = pyglet.shapes.Circle(x=0, y=0, radius=21, color=(0, 0, 0))
        nodeShape = pyglet.shapes.Circle(x=0, y=0, radius=20, color=(255, 255, 255))

        for nodei in range(len(nodePoses)):
            nodeShapeOutline.x = nodePoses[nodei].x
            nodeShapeOutline.y = nodePoses[nodei].y
            nodeShape.x = nodePoses[nodei].x
            nodeShape.y = nodePoses[nodei].y
            label.x = nodePoses[nodei].x
            label.y = nodePoses[nodei].y

            if nodei == self.input_size:
                label.text = "B"
            else:
                label.text = str(nodeNumbers[nodei])

            nodeShapeOutline.draw()
            nodeShape.draw()
            label.draw()

    def JSONstoreNeuralNetwork(self, filepath="NeuralNetwork.json"):
        import jsonpickle
        import jsonpickle.ext.numpy as jsonpickle_numpy
        jsonpickle_numpy.register_handlers()
        jsonpickle.set_encoder_options('json', indent=4)

        with open(filepath, 'w') as file:
            frozen = jsonpickle.encode(self)
            file.write(frozen)

    @staticmethod
    def JSONloadNueralNetwork(filepath="NeuralNetwork.json") -> NeuralNetwork:
        import jsonpickle;
        import jsonpickle.ext.numpy as jsonpickle_numpy;
        jsonpickle_numpy.register_handlers();
        jsonpickle.set_decoder_options('json');

        with open(filepath, 'r') as file:
            templines = file.readlines();

        tempjoined = ''.join(templines);

        thawed = jsonpickle.decode(tempjoined);

        return thawed

#   def serpent_serialize(self):
#   def serpent_deserialize(self, pickledbrain):


#Here because of circular dependancy
class ConnectionHistory:
    def __init__(self, fromNode:int, toNode:int, innovationNumber:int, innovationNumbers:List[int]):
        self.fromNode:int = fromNode
        self.toNode:int = toNode
        self.innovationNumber:int = innovationNumber
        self.innovationNumbers:List[int] = innovationNumbers.copy()

    def __repr__(self):
        return str(self.innovationNumber)

    def matches(self, neuralnetwork:NeuralNetwork, fromNode:Node, toNode:Node) -> bool:
        if len(neuralnetwork.connections) == len(self.innovationNumbers):
            if fromNode.ID == self.fromNode and toNode.ID == self.toNode:
                for coni in range(len(neuralnetwork.connections)):
                    if not neuralnetwork.connections[coni].innovationNumber in self.innovationNumbers:
                        return False
                return True
        return False

def getDuplicateConnections(neuralnetwork:NeuralNetwork):
    existingConnections = []
    duplicateConnections = []
    for connection in neuralnetwork.connections:
        if (connection.fromNode.ID, connection.toNode.ID) in existingConnections:
            duplicateConnections.append((connection.fromNode.ID, connection.toNode.ID))
        else:
            existingConnections.append((connection.fromNode.ID, connection.toNode.ID))
    return duplicateConnections

def printDuplicateConnections(neuralnetwork:NeuralNetwork):
    duplicateconnections = getDuplicateConnections(neuralnetwork)
    log.logger.fatal(duplicateconnections)



if __name__ == "__main__":

    log.logger.info("Starting neuralnetwork.py as main")

    #p = cProfile.Profile()
    #p.runctx('oldbrain.ReLU(x)', locals={'x': 5}, globals={'oldbrain':oldbrain} )
    #p.runcall(oldbrain.fire_network)
    #p.print_stats()


    doonce = True
    if doonce:
        innovationHistory:List[ConnectionHistory] = list()

        log.logger.setLevel(log.logger.INFO)

        ANN1 = NeuralNetwork(9, 9)
        ANN1.generateNetwork()
        log.logger.info("Made ANN1")
        for dummy in range(100):
            ANN1.mutate(innovationHistory)

        output = ANN1.feedForward([1,2,3,4,5,6,7,8,9])
        log.logger.info("ANN1 feedForward: %s" % output)


        ANN1.JSONstoreNeuralNetwork()

        ANN2:NeuralNetwork = NeuralNetwork.JSONloadNueralNetwork()

        output = ANN1.feedForward([1,2,3,4,5,6,7,8,9])
        log.logger.info("ANN1 feedForward: %s" % output)

    log.logger.info("Finished neuralnetwork.py as main")