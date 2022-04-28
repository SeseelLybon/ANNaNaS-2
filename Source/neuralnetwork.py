from __future__ import annotations


from typing import List

import math
import numpy as np
from itertools import combinations
import random

from node import Connection
from node import Node
from maintools import rng

from pymunk import Vec2d

import pyglet
import pyglet.gl as pygl


# NeuralNetwork = Genome


nextConnectionID:int = 1
nextNeuralNetworkID:int = 1

import maintools
log = maintools.colLogger("neuralnetwork")

class NeuralNetwork:

    def __init__(self, input_size:int, output_size:int, ID:int=-1, hollow:bool=False, userecurrency:bool=True):
        global nextNeuralNetworkID
        self.input_size:int = input_size
        self.layers_amount:int = 2
        self.output_size:int = output_size
        self.nextNodeID:int = 0
        self.biasNodeID:int # bias is the last node in self.nodes
        self.useRecurrency = userecurrency;
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

            self.nodes.append(Node(self.nextNodeID))
            self.biasNodeID = self.nextNodeID
            self.nextNodeID+=1
            self.nodes[self.biasNodeID].layer = 0


    def getNode(self, nodeNumber:int):
        for nodei in range(len(self.nodes)):
            if self.nodes[nodei].ID == nodeNumber:
                return self.nodes[nodei]
        return None


    def look(self, vision:List[float]):
        # Set input nodes with vision vector
        for nodei in range(self.input_size):
            self.nodes[nodei].outputValue=vision[nodei]
        self.nodes[self.biasNodeID].outputValue = 1
        pass;

    #def think(self, postClean=False):
    def feedForward(self, vision:List[float], postClean=True) -> List[float]:

        # Set input nodes with vision vector
        for nodei in range(self.input_size):
            self.nodes[nodei].outputValue=vision[nodei]
        self.nodes[self.biasNodeID].outputValue = 1


        for nodei in range(len(self.network)):
            self.network[nodei].fire(self.layers_amount)
            self.network[nodei].inputSum=0;

        outputValues:List[float] = list()
        for outi in range(self.input_size, self.input_size+self.output_size):
            outputValues.append(self.nodes[outi].outputValue)

        if postClean:
            self.resetinputs();

        return outputValues;

    def resetinputs(self):
        for nodei in range(len(self.nodes)):
            self.nodes[nodei].inputSum = 0

    def getDecision(self)->List[float]:
        return [self.nodes[i].outputValue for i in range(self.input_size, self.input_size+self.output_size)];

    def connectNodes(self) -> None:
        for nodei in range(len(self.nodes)):
            self.nodes[nodei].outputConnections.clear()
            self.nodes[nodei].inputConnections.clear()

        for conni in range(len(self.connections)):
            self.connections[conni].fromNode.outputConnections.append(self.connections[conni])
            self.connections[conni].toNode.inputConnections.append(self.connections[conni]);


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
        rnglist = list(range(len(self.connections)))
        rng.shuffle(rnglist)
        randomConnection = None
        for rngcon in rnglist:
            if self.connections[rngcon].enabled:
                if not (self.connections[rngcon].fromNode.ID == self.biasNodeID or self.connections[rngcon].toNode.ID == self.biasNodeID):
                    self.connections[rngcon].enabled = False;
                    randomConnection = rngcon;
                    break;

        # there's no available connection, might as well add one
        if randomConnection is None:
            self.addConnection(innovationHistory)
            return

        # create the new node
        newNodeNumber:int = self.nextNodeID
        self.nodes.append(Node(newNodeNumber))
        self.nextNodeID+=1

        # add new connection with weight 1 from the old connections fromNode to newNode
        connectionInnovationNumber1:int = self.getInnovationNumber(innovationHistory,
                                                                  self.connections[randomConnection].fromNode,
                                                                  self.getNode(newNodeNumber))
        self.connections.append(Connection(self.connections[randomConnection].fromNode,
                                           self.getNode(newNodeNumber),
                                           1,
                                           connectionInnovationNumber1))

        if not self.connections[randomConnection].isRecurrent:
            self.connections[-1].isRecurrent = False

        # add a new connection to the new node with the weight of the disabled connection
        connectionInnovationNumber2 = self.getInnovationNumber(innovationHistory,
                                                              self.getNode(newNodeNumber),
                                                              self.connections[randomConnection].toNode)
        self.connections.append(Connection(self.getNode(newNodeNumber),
                                           self.connections[randomConnection].toNode,
                                           self.connections[randomConnection].weight,
                                           connectionInnovationNumber2))

        if not self.connections[randomConnection].isRecurrent:
            self.getNode(newNodeNumber).layer = self.connections[randomConnection].fromNode.layer + 1
        else:
            self.getNode(newNodeNumber).layer = self.connections[randomConnection].toNode.layer + 1
            self.connections[-1].isRecurrent = True

        # add new connection to the Bias node
        connectionInnovationNumber3 = self.getInnovationNumber(innovationHistory,
                                                               self.getNode(self.biasNodeID),
                                                               self.getNode(newNodeNumber))
        self.connections.append(Connection(self.getNode(self.biasNodeID),
                                           self.getNode(newNodeNumber),
                                           0,
                                           connectionInnovationNumber3 ) )

        # add a new layer if needed
        if not self.connections[randomConnection].isRecurrent:
            if self.getNode(newNodeNumber).layer == self.connections[randomConnection].toNode.layer:
                for nodei in range(len(self.nodes)-1):
                    if self.nodes[nodei].layer >= self.getNode(newNodeNumber).layer:
                        self.nodes[nodei].layer+=1
                self.layers_amount+=1
        else:
            if self.getNode(newNodeNumber).layer == self.connections[randomConnection].fromNode.layer:
                for nodei in range(len(self.nodes)-1):
                    if self.nodes[nodei].layer >= self.getNode(newNodeNumber).layer:
                        self.nodes[nodei].layer+=1
                self.layers_amount+=1

        self.connectNodes()

        log.logger.debug("Added new Node, replacing %d:%d with %d:%d:%d - %s:%s - %d:%d:%d:%d"
                         % (self.connections[randomConnection].fromNode.ID,
                            self.connections[randomConnection].toNode.ID,
                            self.connections[randomConnection].fromNode.ID,
                            newNodeNumber,
                            self.connections[randomConnection].toNode.ID,
                            self.connections[-2].isRecurrent,
                            self.connections[-3].isRecurrent,
                            self.connections[randomConnection].innovationNumber,
                            connectionInnovationNumber1,
                            connectionInnovationNumber2,
                            connectionInnovationNumber3,
                            ))





    def addConnection(self, innovationHistory:List[ConnectionHistory]) -> None:

        if self.useRecurrency and rng.uniform() < 0.10:
            log.logger.debug("Adding new Recurrent Connection")
            isRecurrent = True
        else:
            isRecurrent = False
            log.logger.debug("Adding new Connection")


        #if self.isFullyConnected():
        #    #print("addConnection failed, can't add new connection to filled network")
        #    return

        # TOO: change combinations to permutations to allow nodes to recurrent to themselves
        #   This will break a ton. Also add a visual indicator for the graph

        # grab 2 nodes that don't have a connection
        #if isRecurrent:
        #    rnglist = list( combinations(list(range(len(self.nodes))), 2))+[(i,i) for i in range(self.input_size+2+self.output_size,len(self.nodes))]
        #else:
        #    rnglist = list( combinations(list(range(len(self.nodes))), 2))
        rnglist = list(combinations(list(range(len(self.nodes))), 2));
        rng.shuffle(rnglist)
        randomNode1:int = None
        randomNode2:int = None
        for rngconi in rnglist:
            randomNode1 = rngconi[0]
            randomNode2 = rngconi[1]
            if randomNode1 != self.biasNodeID and randomNode2 != self.biasNodeID:
                if not (self.nodes[randomNode1].layer == 0 and self.nodes[randomNode2].layer == 0):
                    if not (self.nodes[randomNode1].layer == self.layers_amount-1 and self.nodes[randomNode2].layer == self.layers_amount-1):
                        if not self.checkIfConnected(randomNode1, randomNode2, isRecurrent):
                            break;


        #if randomNode1 == self.biasNodeID or randomNode2 == self.biasNodeID:
        #    isRecurrent = False;
        log.logger.debug("%s, %s : these connections were considered non-duplicate" % (randomNode1, randomNode2))

        if not isRecurrent: # got to make sure connection goes forwards
            if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
                temp:int = randomNode2
                randomNode2 = randomNode1
                randomNode1 = temp
        else: # got to make sure connection goes backwards
            if self.nodes[randomNode1].layer < self.nodes[randomNode2].layer:
                temp:int = randomNode2
                randomNode2 = randomNode1
                randomNode1 = temp


        connectionInnovationNumber:int = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        self.connections.append( Connection(self.nodes[randomNode1], self.nodes[randomNode2], rng.normal(), connectionInnovationNumber) )
        self.connectNodes()

        log.logger.debug("Added new Connection %d:%d - %s - %d"
                         % (randomNode1,
                            randomNode2,
                            self.connections[-1].isRecurrent,
                            connectionInnovationNumber))



    def checkIfConnected(self, r1:int, r2:int, isrecurrent:bool) -> bool:
        # if node is itself
        if self.nodes[r1].layer == self.nodes[r2].layer:
            if r1 == r2 and isrecurrent and self.nodes[r1].layer != 0 and self.nodes[r1].layer != self.layers_amount:
                pass;
            else:
                return True

        # If node is connected one or the other way
        for conn in self.connections:
            if (conn.toNode.ID == r1 and conn.fromNode.ID == r2) or \
                    (conn.toNode.ID == r2 and conn.fromNode.ID == r1):
                return True;
        return False

    def getInnovationNumber(self, innovationHistory:List[ConnectionHistory], fromNode:Node, toNode:Node)->int:
        global nextConnectionID
        isNew:bool = True
        connectionInnovationNumber:int = nextConnectionID

        # Check if innovation already exists, then use that innovation
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
        #nodesInLayers:np.ndarray = np.zeros([self.layers_amount], int)
        nodesInLayers:List[int] = [0 for i in range(self.layers_amount)];

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

        # Random thing that makes the population mutate the staler it gets.
        #if staleness == 0:
        #    stalenessMod:float = 1
        #else:
        #    stalenessMod:float = 1+staleness/500

        if rng.uniform() < 0.0005:#*stalenessMod:
            log.logger.debug("RNG: Add new node")
            self.addNode(innovationHistory)

        elif rng.uniform() < 0.005:#*stalenessMod:
            log.logger.debug("RNG: Add new connection")
            self.addConnection(innovationHistory)

        elif rng.uniform() < 0.80:#*stalenessMod:
            log.logger.debug("RNG: Mutate weights")
            for con in random.choices(self.connections, k=len(self.connections)//4):
                con.mutateWeight()
            #if len(self.connections)!=0:
            #    for conni in random.choices(range(len(self.connections)), k=len(self.connections)//8):
            #        self.connections[conni].mutateWeight()

    def crossover(self, parent2:NeuralNetwork) -> NeuralNetwork:
        global nextNeuralNetworkID
        child:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size,
                                            ID=nextNeuralNetworkID,
                                            hollow=True,
                                            userecurrency=self.useRecurrency)
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
        clone:NeuralNetwork = NeuralNetwork(self.input_size, self.output_size, self.ID, hollow=True, userecurrency=self.useRecurrency)
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
                  "\n\tisRecurrent", self.connections[conni].isRecurrent,
                  "\n\tisEnabled", self.connections[conni].enabled,
                  "\n\tfrom layer", self.connections[conni].fromNode.layer,
                  "\n\tto layer", self.connections[conni].toNode.layer,
                  "\n\twith weight:", self.connections[conni].weight)



    def drawNetwork(self, startX:int, startY:int, width:int, height:int, inputlabels:List[str], outputlabels:List[str])->None:
        batch = pyglet.graphics.Batch();
        groupConnections = pyglet.graphics.OrderedGroup(0);
        groupNodesOutlines = pyglet.graphics.OrderedGroup(1);
        groupNodes = pyglet.graphics.OrderedGroup(2);
        groupLabels = pyglet.graphics.OrderedGroup(3);

        shapesList = [];

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
            x:int = int(startX+(layeri*width)/
                        (self.layers_amount + 1.0))
            for nodei in range(len(allNodes[layeri])):
                if layeri == 0:
                    y:int = int(startY + height*(nodei/(len(allNodes[layeri])-1)) );
                elif layeri==self.layers_amount-1:
                    if len(allNodes[layeri]) == 1:
                        y:int = int(startY + height*.5);
                    else:
                        y:int = int(startY + height*(nodei/(len(allNodes[layeri])-1)) );
                else:
                    y:int = int(startY + height*((nodei+1)/(len(allNodes[layeri])+1)) );

                nodePoses.append(Vec2d(x, y))
                nodeNumbers.append(allNodes[layeri][nodei].ID)


        # draw all the connections
        for conni in range(len(self.connections)):
            if self.connections[conni].enabled:
                pygl.glLineWidth(abs(int(self.connections[conni].weight*2))+1)
            else:
                continue

            fromNode_pos:Vec2d = nodePoses[ nodeNumbers.index( self.connections[conni].fromNode.ID ) ]
            toNode_pos:Vec2d = nodePoses[ nodeNumbers.index( self.connections[conni].toNode.ID ) ]

            if not self.connections[conni].isRecurrent:
                if self.connections[conni].weight >=0:
                    col = (255, 0, 0)#, 255, 0, 0,) #red/positive weight
                else:
                    col = (0, 0, 255)#, 0, 0, 255) #blue/negative weight

                shapesList.append( pyglet.shapes.Line(fromNode_pos.x,
                                                      fromNode_pos.y+5,
                                                      toNode_pos.x,
                                                      toNode_pos.y+5,
                                                      width=abs(int(self.connections[conni].weight*2))+1,
                                                      color=col,
                                                      batch=batch,
                                                      group=groupConnections));
            else:
                if self.connections[conni].weight >=0:
                    col = (255, 255, 0) #yellow/positive weight
                else:
                    col = (0, 255, 0) #green/negative weight

                if self.connections[conni].toNode.ID != self.connections[conni].fromNode.ID:
                    shapesList.append( pyglet.shapes.Line(fromNode_pos.x,
                                                          fromNode_pos.y-5,
                                                          toNode_pos.x,
                                                          toNode_pos.y-5,
                                                          width=abs(int(self.connections[conni].weight*2))+1,
                                                          color=col,
                                                          batch=batch,
                                                          group=groupConnections));
                else: # tonode == fromnode
                    # TOO: THIS. IS. GOING. TO. BREAK.
                    #   Past me; you're welcome 0/
                    #if self.connections[conni].toNode.ID == self.connections[conni].fromNode.ID:
                    outlinewidth = abs(int(self.connections[conni].weight*2));
                    shapesList.append(pyglet.shapes.Circle(x=fromNode_pos.x+10, y=fromNode_pos.y+10,
                                                           radius=20+outlinewidth//2+1,
                                                           color=col,
                                                           batch=batch,
                                                           group=groupConnections));
                    shapesList.append(pyglet.shapes.Circle(x=fromNode_pos.x+10, y=fromNode_pos.y+10,
                                                           radius=20-outlinewidth//2,
                                                           color=(178, 178, 178),
                                                           batch=batch,
                                                           group=groupConnections));


        # Draw all nodes (and ID's)
        for nodei in range(len(nodePoses)):
            if self.network[nodei].ID == self.biasNodeID : #self.biasNodeID - this is shifted
                nodeL = "B";
            else:
                nodeL = str(self.network[nodei].ID);
            #nodeL = str(nodeid);
            shapesList.append(pyglet.text.Label(str(nodeL),
                                                font_name='Times New Roman', font_size=14,
                                                x=nodePoses[nodei].x, y=nodePoses[nodei].y,
                                                anchor_x='center', anchor_y='center',
                                                color=(0,0,0, 255),
                                                batch=batch,
                                                group=groupLabels));

            shapesList.append(pyglet.shapes.Circle(x=nodePoses[nodei].x, y=nodePoses[nodei].y,
                                                   radius=21, color=(0, 0, 0),
                                                   batch=batch,
                                                   group=groupNodesOutlines));
            shapesList.append(pyglet.shapes.Circle(x=nodePoses[nodei].x, y=nodePoses[nodei].y,
                                                   radius=20, color=(255, 255, 255),
                                                   batch=batch,
                                                   group=groupNodes));
            #outline.draw();
            #node.draw();

        # make all the input labels
        for labi,nodei in zip(range(len(inputlabels)),range(0, self.input_size)):
            shapesList.append(pyglet.text.Label(inputlabels[labi],
                                                font_name='Times New Roman', font_size=14,
                                                x=nodePoses[nodei].x+30, y=nodePoses[nodei].y,
                                                anchor_x='left', anchor_y='center',
                                                color=(0,0,0, 255),
                                                batch=batch,
                                                group=groupLabels));
        # make all the output labels
        for labi, nodei in zip(range(len(outputlabels)),range(len(self.nodes)-self.output_size, len(self.nodes))):
            shapesList.append(pyglet.text.Label(outputlabels[labi],
                                                font_name='Times New Roman', font_size=14,
                                                x=nodePoses[nodei].x+30, y=nodePoses[nodei].y,
                                                anchor_x='left', anchor_y='center',
                                                color=(0,0,0, 255),
                                                batch=batch,
                                                group=groupLabels));

        batch.draw();
        return;

    def JSONStoreNeuralNetwork(self, filepath="NeuralNetwork.json"):
        import jsonpickle
        import jsonpickle.ext.numpy as jsonpickle_numpy
        jsonpickle_numpy.register_handlers()
        jsonpickle.set_encoder_options('json', indent=4)
        with open(filepath, 'w') as file:
            frozen = jsonpickle.encode(self)
            file.write(frozen)

    @staticmethod
    def JSONLoadNeuralNetwork(filepath="NeuralNetwork.json") -> NeuralNetwork:
        import jsonpickle;
        import jsonpickle.ext.numpy as jsonpickle_numpy;
        jsonpickle_numpy.register_handlers();
        jsonpickle.set_decoder_options('json');
        with open(filepath, 'r') as file:
            templines = file.readlines();
        tempjoined = ''.join(templines);
        thawed = jsonpickle.decode(tempjoined);
        return thawed


    def train(self, training_data:list, training_answers:list, learnrate:float=0.01)-> None:
        assert len(training_data)==len(training_answers);
        deltaimages:List[dict] = []
        # create deltaimages
        # A list of all deltaimages created.
        for t_d, t_a in zip(training_data, training_answers):
          deltaimages.append(self.createGDDeltaImage(t_d, t_a));

        # combine all deltaimages into a summed image
        finalDeltaImage = { con.innovationNumber:0 for con in self.connections};
        for deltaimage in deltaimages:
            for k in finalDeltaImage.keys():
                finalDeltaImage[k] += deltaimage[k];

        # make an average of the summed image
        for k in finalDeltaImage.keys():
            finalDeltaImage[k] /= len(deltaimages);

        # apply finalDeltaImage
        self.applyGDDeltaImage(learnrate, finalDeltaImage)
        pass;


    def createGDDeltaImage(self, training_input, training_answers) -> dict:
        #if len(training_input) != self.input_size or len(training_answers)!= self.output_size:
        #    log.logger.fatal("training input(%d) and output(%d) are note the same!"%(len(training_input), len(training_answers)));
        #    assert False;

        deltaimage:dict = { con.innovationNumber:0 for con in self.connections};

        self.feedForward(training_input, postClean=False);

        decision = self.getDecision();

        # create stuff to keep track
        # list of nudges per connection to be applied later
        connectiongNudgeDict:dict = { con.innovationNumber:[] for con in self.connections};

        #a_L-1
        nodeNudgeDict = {node.ID:0 for node in self.network};

        # start with appling the output error
        for i in range(len(self.network)-1, len(self.network)-self.output_size-1, -1):

            for con in self.network[i].inputConnections:
                a_L1 = con.fromNode.outputValue;
                w_L = con.weight;
                d_a_L = 2*(con.toNode.outputValue-training_answers[len(self.network)-i-1]);

                connectiongNudgeDict[con.innovationNumber].append( d_a_L * -1 * a_L1);
                nodeNudgeDict[i] += d_a_L * w_L * a_L1;

        # make delta image

        # In reverse - as this is backprop
        for nodi in range(len(self.network)-self.output_size-1, 0-1, -1):
            #if self.network[nodi].layer == 0 or self.network[nodi].layer == self.layers_amount:
            #    # just a lazy way to surcomvent setting the range() correctly
            #    continue;

            for con in self.network[nodi].inputConnections:
                a_L1 = con.fromNode.outputValue;
                w_L = con.weight;
                d_a_L = 2*(con.toNode.outputValue-nodeNudgeDict[con.toNode.ID]); #

                connectiongNudgeDict[con.innovationNumber].append( a_L1 * 1 * d_a_L );
                nodeNudgeDict[nodi] += a_L1 * w_L * d_a_L;


        for coniNu in connectiongNudgeDict.keys():
            for nudge in connectiongNudgeDict[coniNu]:
                if len(connectiongNudgeDict[coniNu]) >0:
                    deltaimage[coniNu] = nudge/len(connectiongNudgeDict[coniNu]);

        self.feedForward(training_input, postClean=True);

        return deltaimage;

    def applyGDDeltaImage(self, learnrate:float, deltaimage:dict)->None:
        for con in self.connections:
            #con.weight += deltaimage[con.innovationNumber]*learnrate;
            con.weight += min(4,max(-4,deltaimage[con.innovationNumber]*learnrate));

    def GDSigmoid(self,x):
        return 1/(1+math.e**-x);

    def costfunction(self, correct_output:list)->float:
        assert len(correct_output) == self.output_size;

        total=0
        for cor, out in zip(correct_output, self.getDecision()):
            total+= (out - cor)**2

        return total

#Here because of circular dependancy
class ConnectionHistory:
    def __init__(self, fromNode:int, toNode:int, innovationNumber:int, innovationNumbers:List[int]):
        self.fromNode:int = fromNode
        self.toNode:int = toNode
        self.innovationNumber:int = innovationNumber
        self.innovationNumbers:List[int] = innovationNumbers#.copy()

    def __repr__(self):
        return str(self.innovationNumber)

    def matches(self, neuralnetwork:NeuralNetwork, fromNode:Node, toNode:Node) -> bool:
        # TODO: Still causes duplicates
        # There are more connections than innovation numbers in a single ANN.
        if len(neuralnetwork.connections) == len(self.innovationNumbers):
            if fromNode.ID == self.fromNode and toNode.ID == self.toNode:
                for coni in range(len(neuralnetwork.connections)):
                    if not neuralnetwork.connections[coni].innovationNumber in self.innovationNumbers:
                        return False;
                return True;
            return False;
        return False;

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


import unittest
class TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from logging import DEBUG
        from logging import INFO
        log.logger.setLevel(DEBUG);
        from numpy.random import default_rng
        #from maintools import rng
        #cls.rng = default_rng(11037)

    def tearDown(self)->None:
        pass;

    #@unittest.expectedFailure
    #def functioniexpecttofail(self):
    #   pass;

    def test_basebehaviour(self):
        with self.subTest("basic creation"):
            testinnovationHistory = list();
            testANN = NeuralNetwork(3,3)
            #for i in range(10000):
            #    testANN.mutate(testinnovationHistory);
            for u in range(20):
                for j in range(10):
                    testANN.addConnection(testinnovationHistory);
                for j in range(10):
                    testANN.addNode(testinnovationHistory);
            testANN.generateNetwork();
            #testANN.printNetwork();
            #testANN.generateNetwork();

        with self.subTest("No duplicate connections"):
            for con_a in testANN.connections:
                paira = (con_a.toNode.ID, con_a.fromNode.ID);
                #appearedoncealready = False;
                for con_b in testANN.connections:
                    pairb = (con_b.toNode.ID, con_b.fromNode.ID);
                    if paira[0] == pairb[0] and paira[1] == pairb[1]:
                        #if not appearedoncealready:
                        with self.subTest("No duplicate connections"):
                            if not (con_a.enabled and con_b.enabled) or con_a.innovationNumber == con_b.innovationNumber:
                                # this is probably itself
                                #appearedoncealready = True;
                                pass;
                            else:
                                self.assertTrue(False,"Found duplicate: %d:%d %d:%d"%(paira[1],paira[0], con_a.innovationNumber, con_b.innovationNumber));

        with self.subTest("Feed forward w/o recurrency"):
            traindata = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]];
            trainanswers = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]];

            errorrate1 = 0;
            for dat, ans in zip(traindata, trainanswers):
                testANN.feedForward(dat);
                deci = testANN.getDecision();
                log.logger.info(deci);
                for i in range(len(deci)):
                    errorrate1 += (deci[i]-ans[i])**2
            log.logger.info(errorrate1);

            # expecting the end values to be none 0's.
            self.assertTrue(testANN.getDecision() != [0,0,0])
            errorrate2 = 0;
            for dat, ans in zip(traindata, trainanswers):
                testANN.feedForward(dat);
                deci = testANN.getDecision();
                log.logger.info(deci);
                for i in range(len(deci)):
                    errorrate2 += (deci[i]-ans[i])**2
            log.logger.info(errorrate2);

            #Expecting the two values to be equal since no change should be happening between runs
            self.assertTrue(errorrate1 == errorrate2);

        with self.subTest("Feed forward w/ recurrency"):
            errorrate1 = 0;
            for dat, ans in zip(traindata, trainanswers):
                testANN.feedForward(dat, postClean=False);
                deci = testANN.getDecision();
                log.logger.info(deci);
                for i in range(len(deci)):
                    errorrate1 += (deci[i]-ans[i])**2
            log.logger.info(errorrate1);

            errorrate2 = 0;
            for dat, ans in zip(traindata, trainanswers):
                testANN.feedForward(dat, postClean=False);
                deci = testANN.getDecision();
                log.logger.info(deci);
                for i in range(len(deci)):
                    errorrate2 += (deci[i]-ans[i])**2
            log.logger.info(errorrate2);

            # Expecting the two values to be unequal since the recurrency must influence the second feedforward.
            self.assertFalse(errorrate1 == errorrate2);


        return;

    @unittest.skip("not testing test_Graphics, test expensive")
    def test_Graphics(self):
        import time;
        windowMain = pyglet.window.Window(1200, 800)
        testinnovationHistory = list();
        testANN = NeuralNetwork(3,3)
        for i in range(5000):
            testANN.mutate(testinnovationHistory);
        testANN.generateNetwork();
        starttime = time.time();

        def update(dt):
            windowMain.clear()
            pyglet.gl.glClearColor(0.7,0.7,0.7,1)
            testANN.drawNetwork(50, 50, 1100, 650, ["A","B","C"], ["D","E","F"]);
            if time.time() > starttime + 20:
                pyglet.app.exit();
            return;

        pyglet.clock.schedule_interval_soft(update, 1)
        pyglet.app.run()
        pass;


    #@unittest.skip("not testing test_Train")
    #@unittest.expectedFailure
    def test_Train(self):
        testinnovationHistory = list();
        testANN = NeuralNetwork(3,3)
        for i in range(2000):
            testANN.mutate(testinnovationHistory);
        testANN.generateNetwork();
        traindata = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]];
        trainanswers = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]];

        errorrate1 = 0;
        for dat, ans in zip(traindata, trainanswers):
            testANN.feedForward(dat);
            deci = testANN.getDecision();
            log.logger.info(deci);
            for i in range(len(deci)):
                errorrate1 += (deci[i]-ans[i])**2
        log.logger.info(errorrate1);
        for i in range(10):
            testANN.train(traindata, trainanswers, 0.01);

        errorrate2 = 0;
        for dat, ans in zip(traindata, trainanswers):
            testANN.feedForward(dat);
            deci = testANN.getDecision();
            log.logger.info(deci);
            for i in range(len(deci)):
                errorrate2 += (deci[i]-ans[i])**2
        log.logger.info(errorrate2);

        self.assertTrue(errorrate1 > errorrate2);
        pass;

    #@unittest.skip("not testing test_createGDimage")
    #@unittest.expectedFailure
    def test_createGDimage(self):
        testinnovationHistory = list();
        testANN = NeuralNetwork(3,3)
        for i in range(100):
            testANN.mutate(testinnovationHistory);
        testANN.generateNetwork();

        DI = testANN.createGDDeltaImage([1,1,1],[1,1,1]);
        #print(DI)
        # Cannot assert

    @unittest.skip("not testing test_applyGDimage")
    @unittest.expectedFailure
    def test_applyGDimage(self):
        testinnovationHistory = list();
        testANN = NeuralNetwork(3,3)
        for i in range(100):
            testANN.mutate(testinnovationHistory);
        testANN.generateNetwork();

        DI = testANN.createGDDeltaImage([1,1,1],[1,1,1]);
        #print(DI)
        preweights =[con.weight for con in testANN.connections]
        #print(preweights);
        testANN.applyGDDeltaImage(1, DI);
        postweights = [con.weight for con in testANN.connections]
        #print(postweights);
        self.assertFalse(preweights==postweights);

        postweightschange = [pre-post for pre, post in zip(postweights, preweights)];
        #print(sorted(postweightschange))
        #print(sorted(DI.values()))
        #self.assertTrue(sorted(DI.values())==sorted(postweightschange));
        for post, di in zip(sorted(DI.values()),sorted(postweightschange)):
            self.assertAlmostEqual(post, di,
                                   msg="%f - %f"%(post, di));