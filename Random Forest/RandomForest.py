import math
import random
from datetime import datetime

# A forest made of trees
class Forest:
    # Initialise Forest
    def __init__(self):
        self.trees = []

    # Grow trees in forest
    def plantTrees(self, forestSize, features, trainingData):
        print "Building " + str(forestSize) + " trees."
        for x in range(forestSize):
            # Create Tree
            newTree = Tree()
            # Bootstrap Sample of dataSet
            bootstrapSample = []
            m = int(len(trainingData)*0.66)
            for x in range(m):
                bootstrapSample.append(random.choice(trainingData))
            # Grow Tree
            newTree.grow(newTree.rootNode, features, bootstrapSample)
            self.trees.append(newTree)

    # Query Forest with data point
    def queryForest(self, inputArray):
        estimate = [0, 0, 0, 0, 0]
        # Query each tree
        for tree in self.trees:
            probDist = tree.traverse(inputArray)
            for x in range(5):
                estimate[x] += probDist[x]
        # Aggregate Prediction Scores
        for x in range(5):
            estimate[x] = estimate[x]/len(self.trees)
        return estimate


# One Classification Tree
class Tree:
    # Initialise Tree
    def __init__(self):
        self.rootNode = Node(0)

    # Traverse tree to obtain prediction
    def traverse(self, inputArray):
        # Start at root
        currNode = self.rootNode
        while (len(currNode.children) > 0):
            # Go Left or Right based on split at node
            if (inputArray[currNode.splitFeature] > currNode.splitValue):
                currNode = currNode.children[1]
            else:
                currNode = currNode.children[0]
        return currNode.probDist

    # Grow Tree
    def grow(self, currNode, parameters, dataSet):
        D = currNode.depth
        # Select 4 random parameters
        rndParams = random.sample(parameters, 4)
        # Find best separator
        currNode.findBestSeparator(rndParams, dataSet)
        # if split is found, create and expand children
        if (currNode.splitValue != 0):
            # Create children
            currNode.children = [Node(D+1), Node(D+1)]
            # Grow Children recursively
            childSetL, childSetR = splitData(dataSet, currNode.splitFeature, currNode.splitValue)
            self.grow(currNode.children[0], parameters, childSetL)
            self.grow(currNode.children[1], parameters, childSetR)


# One Node of a Tree
class Node:
    # Initialise Node
    def __init__(self, D):
        self.children = ""
        self.splitFeature = 0
        self.splitValue = 0
        self.probDist = []
        self.depth = D

    # Find best separator
    def findBestSeparator(self, parameters, dataSet):
        # Initialise Variables
        bestParam = ""
        bestSepVal = 0
        bestIG = 0
        # Try for each parameter
        for param in parameters:
            # Test for split at evenly spaced data points (max 100)
            splitSet = []
            if len(dataSet) > 100:
                step = len(dataSet)/100
                for index in range(step, len(dataSet), step):
                    splitSet.append(dataSet[index])
            else:
                splitSet = dataSet
            for each in splitSet:
                splitVal = each[param]
                childSetL, childSetR = splitData(dataSet, param, splitVal)
                # Check child sets are larger than 10
                if ((len(childSetL) > 10) and (len(childSetR) > 10)):
                    infoGain = informationGain(dataSet, childSetL, childSetR)
                    # New Best Split
                    if (infoGain > bestIG):
                        bestIG = infoGain
                        bestSepVal = splitVal
                        bestParam = param
                else:
                    self.probDist = classProb(countClasses(dataSet))
        self.splitValue = bestSepVal
        self.splitFeature = bestParam

# Get Data From file
def getData(filename, trainingRun):
    readfile = open(filename, 'r')

    # Parse all data into an array of Dates and Floats.
    testData = []
    for line in readfile:
        line = line.strip()
        line = line.split("\t")
        for x in range(1, len(line)-1):
            line[x] = float(line[x])
        line[17] = int(line[17])
        testData.append(line)
    readfile.close()

    # Discard data put aside for training other methods.
    if (trainingRun):
        testData = testData[:1934]
    else:
        testData = testData[1934:]
    return testData

# Split data to the left and right of a given value for a given feature.
def splitData(dataSet, splitFeature, splitValue):
    childSetL = []
    childSetR = []
    for point in dataSet:
        if (point[splitFeature] > splitValue):
            childSetR.append(point)
        else:
            childSetL.append(point)
    return childSetL, childSetR

# Count numbers of each class in dataSet
def countClasses(dataSet):
    classCount = [0, 0, 0, 0, 0]
    for point in dataSet:
        classCount[point[17]] += 1
    return classCount

# return probability matrix from classcount structure
def classProb(classCount):
    n = sum(classCount)
    classCount = [(float(x)/n) for x in classCount]
    return classCount

# calculate shannon entropy from class probability struct
def calcEntropy(classProbArray):
    entropy = 0
    for c in classProbArray:
        if (c != 0):
            entropy += c*math.log(c, 2)
    entropy = -entropy
    return entropy

# Calculate Information gain of split
def informationGain(dataSet, childSetL, childSetR):
    # Get class probability distributions
    classProbArray = [classProb(countClasses(dataSet)), classProb(countClasses(childSetL)), classProb(countClasses(childSetR))]
    # calculate Shannon Entropies
    EntropyArray = [calcEntropy(each) for each in classProbArray]
    # Calculate information Gain
    x = (len(childSetL)/len(dataSet))*EntropyArray[1]
    x += (len(childSetR)/len(dataSet))*EntropyArray[2]
    infoGain = EntropyArray[0] - x
    return infoGain

# Config
filename = "1BA.txt"
forestSize = 100
PP = 1
features = [x for x in range(2, 16)]

# Evaluation statistics
nInvMade = 0
invMade = 0
returnMade = 0
nAsc = 0
nDesc = 0
nCorr = 0
nAscCorr = 0
predictions = []

# Build Forest
start = datetime.now()
forest = Forest()
trainingData = getData(filename, 1)
forest.plantTrees(forestSize, features, trainingData)
print datetime.now()-start

# Test forest
testData = getData(filename, 0)
# For each test case
for x in range(len(testData)-PP):
    # Get Forest Estimate, actual class
    estimate = forest.queryForest(testData[x])
    prob = max(estimate)
    estimate = estimate.index(max(estimate))
    predictions.append(estimate)
    actualClass = testData[x][17]
    # determine if investment to be made, and resolve
    if (estimate == 0):
        nInvMade += 1
        invMade += 2*testData[x][1]
        returnMade += 2*testData[x+PP][1]
    elif (estimate == 1):
        nInvMade += 1
        invMade += testData[x][1]
        returnMade += testData[x+PP][1]
    # Accuracy Calculations
    if (actualClass < 2):
        nAsc += 1
        if (estimate <= actualClass):
            nCorr += 1
            nAscCorr += 1
    else:
        if (estimate >= actualClass):
            nCorr +=1
        nDesc += 1


# Calculate Evaluation statistics
AscCorr = (float(nAscCorr)/float(nAsc))*100
CorrRatio = (float(nCorr)/(float(nAsc)+float(nDesc)))*100
if nInvMade > 0:
    ROI = returnMade/invMade
print "Correct %: " + str(CorrRatio)
print "Profitable Correct: " + str(AscCorr)
print "Number of investments: " + str(nInvMade)
print "Investment: " + str(invMade)
print "Return: " + str(returnMade)
print "ROI: " + str(ROI)
print predictions.count(0), predictions.count(1), predictions.count(2), predictions.count(3), predictions.count(4)
