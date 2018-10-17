import random
import numpy
import math
from abc import ABC, abstractmethod
import solver
from line_profiler import LineProfiler

class SimilarityCalculator(ABC):

    def calculateWeight(self, attribute1, attribute2):
        pass

    def calculateCentroidWithNewNode(self, cluster, newNode):
        pass

    def calculateCentroidWithDeletedNode(self, cluster, deletedNode):
        pass

class EuclideanSimilarityCalculator(SimilarityCalculator):

    def calculateWeight(self, attribute1, attribute2):
        #dist = [(a - b)**2 for a, b in zip(attribute1, attribute2)]
        #dist = math.sqrt(sum(dist))
        #return dist
        attribute1 = numpy.asarray(attribute1)
        attribute2 = numpy.asarray(attribute2)
        return numpy.linalg.norm(attribute1-attribute2)

    def calculateCentroidWithNewNode(self, cluster, newNode):
        attributes = [0]*len(newNode.attributes)
        for node in cluster.getNodes():
            for i in range(0, len(attributes)):
                attributes[i] = attributes[i] + node.attributes[i]
        for i in range(0, len(attributes)):
            attributes[i] = attributes[i]/cluster.size
        return attributes

    def calculateCentroidWithDeletedNode(self, cluster, deletedNode):
        pass

class ImprovedEuclideanSimilarityCalculator(EuclideanSimilarityCalculator):

    def calculateWeight(self, attribute1, attribute2):
        temp1 = attribute1[0]-attribute2[0]
        temp2 = attribute1[1]-attribute2[1]
        return (temp1*temp1+temp2*temp2)**0.5

    def calculateCentroidWithNewNode(self, cluster, newNode):
        attributes = [0]*len(newNode.attributes)
        if cluster.size == 1:
            attributes = newNode.attributes.copy()
            return attributes
        for i in range(0, len(attributes)):
            attributes[i] = (cluster.centroid[i]*(cluster.size-1)+newNode.attributes[i])/cluster.size
        return attributes

    def calculateCentroidWithDeletedNode(self, cluster, deletedNode):
        attributes = [0]*len(deletedNode.attributes)
        if cluster.size == 0:
            return -1
        for i in range(0, len(attributes)):
            attributes[i] = (cluster.centroid[i]*(cluster.size+1)-deletedNode.attributes[i])/cluster.size
        return attributes


class Node:

    def __init__(self, id, attributes):
        self.id = id
        self.clustersPheromone = 0
        self.attributes = attributes

class Cluster:

    def __init__(self, similarityCalculator):
        self.size = 0
        self.nodes = []
        self.centroid = -1
        self.similarityCalculator = similarityCalculator
        self.modified = False
        self.temp = []

    def updateCentroidWithNewNode(self, newNode):
        self.centroid = self.similarityCalculator.calculateCentroidWithNewNode(self, newNode)

    def updateCentroidWithDeletedNode(self, deletedNode):
        self.centroid = self.similarityCalculator.calculateCentroidWithDeletedNode(self, deletedNode)

    def addNode(self, node):
        self.modified = True
        self.size = self.size + 1
        self.nodes.append(node)
        self.updateCentroidWithNewNode(node)

    def removeNode(self, pos):
        self.modified = True
        self.size = self.size - 1
        temp = self.nodes[pos]
        self.nodes[pos] = None
        self.updateCentroidWithDeletedNode(temp)
        
    def getNodes(self):
        if self.modified == False:
            return self.temp
        self.modified = False
        self.temp = []
        for node in self.nodes:
            if node != None:
                self.temp.append(node)
        return self.temp
     
class TransitionRule(ABC):

    def getCluster(self, node, centroids):
        pass

class AntTransitionRule(TransitionRule):

    def __init__(self, a, b, q0, similarityCalculator):
        self.a = a
        self.b = b
        self.q0 = q0
        self.similarityCalculator = similarityCalculator
        self.calculateWeight = similarityCalculator.calculateWeight
    
    def getCluster(self, node, clusters):
        prob = [0]*len(clusters)
        cumSum = 0
        for i in range(0, len(clusters)):
            if clusters[i].size == 0:
                relativeWeight = 1
            else:
                relativeWeight = 1/self.calculateWeight(node.attributes, clusters[i].centroid)
            prob[i] = (relativeWeight**self.b)*(node.clustersPheromone[i]**self.a)
            cumSum = cumSum + prob[i]
        exploitationRes = 0
        exploitationMax = 0
        for i in range(0, len(clusters)):
            prob[i] = prob[i]/cumSum
            if prob[i] > exploitationMax:
                exploitationMax = prob[i]
                exploitationRes = i
        q = numpy.random.random_sample()
        if q <= self.q0:
            return exploitationRes
        else:
            return numpy.random.choice(len(clusters), 1, prob)[0]

class Ant:

    def __init__(self, clustersNumber, transitionRule):
        self.clusters = [Cluster(transitionRule.similarityCalculator) for c in range(0, clustersNumber)]
        self.transitionRule = transitionRule

    def move(self, nodes):
        shuffledNodes = nodes.copy()
        random.shuffle(shuffledNodes)
        for node in shuffledNodes:
            cluster = self.transitionRule.getCluster(node, self.clusters)
            self.clusters[cluster].addNode(node)
            
class Graph:

    def __init__(self, size):
            self.nodes = [None]*size

    def addNode(self, node):
        self.nodes[node.id] = node

class ACOC:

    def calculateObjectiveFunction(self, clusters, similarityCalculator):
        f = 0
        for cluster in clusters:
            for node in cluster.nodes:
                f = f + similarityCalculator.calculateWeight(node.attributes, cluster.centroid)
        return f

    def updatePheromone(self, quality, clustering, p, Q):
        for i in range(0, len(clustering)):
            for node in clustering[i].getNodes():
                for j in range(0, len(node.clustersPheromone)):
                    node.clustersPheromone[j] = node.clustersPheromone[j]*(1-p)
                    if i==j:
                        node.clustersPheromone[j] += Q/quality

    def localSearch(self, clusters, similarityCalculator):
        nodes = [None]*len(clusters)
        for c in range(0, len(clusters)):
            nodes[c] = clusters[c].getNodes()
        for c in range(0, len(clusters)):   
            for n in range(0, len(nodes[c])):
                minimum = similarityCalculator.calculateWeight(clusters[c].centroid, nodes[c][n].attributes)
                best = c
                for c1 in range(0, len(clusters)):
                    if c1 != c:
                        if clusters[c1].size == 0:
                            minimum = 0
                            best = c1
                        else:
                            temp = similarityCalculator.calculateWeight(clusters[c1].centroid, nodes[c][n].attributes)
                            if temp < minimum:
                                minimum = temp
                                best = c1
                if best != c:
                    clusters[c].removeNode(n)
                    clusters[best].addNode(nodes[c][n])

    def run(self, clustersNumber, a, b, q0, t01, t02, p, Q, iterations, antsNumber, graph, similarityCalculator):
        transitionRule = AntTransitionRule(a, b, q0, similarityCalculator)
        for node in graph.nodes:
            node.clustersPheromone = [(t02-t01)*numpy.random.random_sample()+t01]*clustersNumber
        bestClustering = (None, None)
        while iterations>0:
            iterations = iterations-1
            ants = [Ant(clustersNumber, transitionRule) for a in range(0, antsNumber)]
            tempBestClustering = (None, None)
            for ant in ants:
                ant.move(graph.nodes)
                temp = (self.calculateObjectiveFunction(ant.clusters, similarityCalculator), ant.clusters)
                if tempBestClustering[0] == None or temp[0] < tempBestClustering[0]:
                    tempBestClustering = temp
            self.localSearch(tempBestClustering[1], similarityCalculator)
            if bestClustering[0] == None or tempBestClustering[0] < bestClustering[0]:
                bestClustering = tempBestClustering
            self.updatePheromone(tempBestClustering[0], tempBestClustering[1], p, Q)
        print("Best error: "+str(bestClustering[0]))
        return bestClustering[1]

class ACOCsolver(solver.Solver):

    def solve(self, objects, clustersNumber):
        graph = Graph(len(objects))
        cont = 0
        for o in objects:
            graph.addNode(Node(cont, list(o)))
            cont = cont+1
        clusters = ACOC().run(clustersNumber, 1, 6, 0.8, 0.99, 1.01, 0.6, 5, 80, 12, graph, ImprovedEuclideanSimilarityCalculator())
        for c in range(0, len(clusters)):
            nodes = clusters[c].getNodes()
            for i in range(0, len(nodes)):
                nodes[i] = nodes[i].attributes
            clusters[c] = nodes
        return clusters
