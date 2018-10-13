import random
import numpy
from abc import ABC, abstractmethod
import solver

class SimilarityCalculator(ABC):

    def calculateWeight(self, attribute1, attribute2):
        pass

    def calculateCentroid(self, cluster, newNode):
        pass

class EuclideanSimilarityCalculator(SimilarityCalculator):

    def calculateWeight(self, attribute1, attribute2):
        return numpy.linalg.norm(attribute1-attribute2)

    def calculateCentroid(self, cluster, newNode):
        attributes = [0]*len(newNode.attributes)
        for node in cluster.nodes:
            for i in range(0, len(attributes)):
                attributes[i] = attributes[i] + node.attributes[i]
        for i in range(0, len(attributes)):
            attributes[i] = attributes[i]/len(cluster)
        return attributes

class Node:

    def __init__(self, id, clustersPheromone, attributes):
        self.id = id
        self.clustersPheromone = clustersPheromone
        self.attributes = attributes

class Cluster:

    def __init__(self, similarityCalculator):
        self.nodes = []
        self.centroid = -1
        self.similarityCalculator = similarityCalculator

    def updateCentroid(self, newNode):
        self.centroid = self.similarityCalculator.calculateCentroid(self, newNode)

    def addNode(self, node):
        self.nodes.append(node)
        self.updateCentroid(node)

    def isEmpty(self):
        if len(self.nodes) == 0:
            return True
        else:
            return False
     
class TransitionRule(ABC):

    def getCluster(self, node, centroids):
        pass

class AntTransitionRule(TransitionRule):

    def __init__(self, a, b, q0, similarityCalculator):
        self.a = a
        self.b = b
        self.q0 = q0
        self.similarityCalculator = similarityCalculator

    def getCluster(self, node, clusters):
        prob = []
        cumSum = 0
        for i in range(0, len(clusters)):
            if clusters[i].isEmpty() == True:
                relativeWeight = 1
            else:
                relativeWeight = 1/self.similarityCalculator.calculateWeight(node.attributes, clusters[i].centroid)
            prob[i] = pow(relativeWeight, self.b)*pow(node.clustersPheromone[i], self.a)
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
        self.clusters = [Cluster(transitionRule.similarityCalculator)]*clustersNumber
        self.transitionRule = transitionRule

    def move(self, nodes):
        nodes = nodes.copy()
        random.shuffle(nodes)
        while len(nodes) > 0:
            currentNode = nodes.remove(0)
            cluster = self.transitionRule.getCluster(currentNode, self.clusters)
            self.clusters[cluster].addNode(currentNode)
            
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

    def updatePheromone(self, quality, clustering, nodes, p):
        for i in range(0, len(clustering)):
            for node in clustering[i].nodes:
                for j in range(0, len(node.clustersPheromone)):
                    node.clustersPheromone[j] = node.clustersPheromone[j]*(1-p)
                    if i==j:
                        node.clustersPheromone[j] += 1/quality
                    
    def run(self, clustersNumber, a, b, q0, t01, t02, p, iterations, antsNumber, eliteAntsNumber, graph, similarityCalculator):
        transitionRule = AntTransitionRule(a, b, q0, similarityCalculator)
        for node in graph.nodes:
            node.clustersPheromone = [(t02-t01)*numpy.random.random_sample()+t01]*clustersNumber
        bestClustering = (None, None)
        while iterations>0:
            iterations = iterations-1
            ants = [Ant(clustersNumber, transitionRule)]* antsNumber
            clusterings = []
            for ant in ants:
                ant.move(graph.nodes)
                clusterings.append((self.calculateObjectiveFunction(ant.clusters, similarityCalculator), ant.clusters))
            clusterings.sort()
            if bestClustering[0] == None or clusterings[0][0] < bestClustering[0]:
                bestClustering = clusterings[0]
            for r in range(0, eliteAntsNumber):
                self.updatePheromone(clusterings[r][0], clusterings[r][1], graph.nodes, p)

class ACOCsolver(solver.Solver):

    def solve(self, objects):
        graph = Graph(len(objects))
        cont = 0
        for o in objects:
            graph.addNode(Node(cont, o[0], o[1]))
            cont = cont+1
        ACOC().run(4, 1, 2, 0.0001, 0.7, 0.8, 0.1, 1000, 10, 1, graph, EuclideanSimilarityCalculator())
