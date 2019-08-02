import numpy as np
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime

class MultiLayerPerceptron:
    #Base Class for MultiLayerPerceptron

    def __init__(self, num_Input, num_Hidden, num_Output):
        self.num_Input = num_Input
        self.num_Hidden = num_Hidden
        self.num_Output = num_Output

        self.learning_Rate = 0.01
        self.a = float(2)/self.num_Input
        self.b = float(2)/self.num_Hidden

        self.weights_IH = np.random.uniform(-self.a, self.a, (self.num_Hidden, self.num_Input))
        self.weights_HO = np.random.uniform(-self.b, self.b, (self.num_Output, self.num_Hidden))

        self.bias_H = np.zeros([num_Hidden, 1])
        self.bias_O = np.zeros([num_Output, 1])

    def predict(self, input_Array):
        # takes array of inputs, outputs output layer as matrix.
        input_Matrix = np.asmatrix(input_Array).T

        # Hidden Weights
        hidden = self.weights_IH.dot(input_Matrix)
        hidden = np.add(self.bias_H, hidden)

        # Activation Function
        hidden = sigmoidMatrix(hidden)

        output = self.weights_HO.dot(hidden)
        output = np.add(self.bias_O, output)
        output = sigmoidMatrix(output)

        return float(output)

    def train(self, input_Array, target):
        # Run Feed Forward
        input_Matrix = np.asmatrix(input_Array)

        # Hidden Weights
        hidden = self.weights_IH.dot(input_Matrix.T)
        hidden = np.add(self.bias_H, hidden)

        # Activation Function
        sigmoidMatrix(hidden)

        output = self.weights_HO.dot(hidden)
        output = np.add(self.bias_O, output)
        sigmoidMatrix(output)

        # Output Error
        output_Error = target - output

        # Gradient calculation - HO layer
        gradient = (output * (1 - output))
        gradient = gradient * output_Error * self.learning_Rate
        weights_HO_delta = gradient.dot(hidden.T)

        self.weights_HO = self.weights_HO + weights_HO_delta
        self.bias_O = self.bias_O + gradient

        # Hidden Errors
        hidden_Errors = (self.weights_HO.T).dot(output_Error)

        # Gradient Calculation - IH Layer
        gradient = derivativeSigmoid(hidden)
        hidden_Errors = np.array(hidden_Errors)
        gradient = np.array(gradient)
        gradient = gradient * hidden_Errors * self.learning_Rate
        gradient = np.matrix(gradient)
        weights_IH_delta = gradient.dot(input_Matrix)

        self.weights_IH = self.weights_IH + weights_IH_delta
        self.bias_H = np.add(self.bias_H, gradient)

        # Get % error for threshold Algorithm
        errPercent = math.fabs(output - target)/target

        return float(errPercent), float(output)

    # Train MLP using data from tab delimited file.
    def beginTraining(self, PP, training_Data, zValue):

        # Train Model
        errArray = []
        aChange = []
        sError = 0
        nTests = len(training_Data)-PP
        for index in range(nTests):
            target = training_Data[index+PP][1]/training_Data[index][17]
            target = target * 0.6
            target += 0.2
            a, b = self.train(training_Data[index][2:17], target)
            errArray.append(a)
            sError += (a)

            # Actual % change
            lstObsrv = training_Data[index][15]
            actual = training_Data[index+PP][1]/training_Data[index][17]
            aChange.append(((lstObsrv-actual)/lstObsrv)*100)

        aPlot, = plt.plot(np.arange(len(errArray)), errArray)
        plt.legend([aPlot], ["% Error (/100)"], loc=7)
        plt.title("Number of Hidden Nodes: " + str(self.num_Hidden))
        plt.xlabel("Epochs")
        #plt.show()

        # Calculations for trading threshold algorithm
        avgError = sError/nTests
        sErrorSquared = 0
        for error in errArray:
            sErrorSquared += (error - avgError)**2
        stdDev = math.sqrt(sErrorSquared / (nTests-1))
        upperBound = avgError + zValue * stdDev
        print("avgError: " + str(avgError))
        print("stdDev: " + str(stdDev))
        print("Upper Error Bound: " + str(upperBound))

        return upperBound

    def testModel(self, PP, test_Data, upperBound):
        nInvMade = 0
        invMade = 0
        returnMade = 0
        nAsc = 0
        nDesc = 0
        nCorr = 0
        nAscCorr = 0
        nInvCorr = 0
        inputs, outputs, targets, changePerc, aChange = [], [], [], [], []

        nTests = len(test_Data)-PP
        for index in range(nTests):
            # output normalised value
            out = self.predict(test_Data[index][2:17])
            # Obtain real value of output
            out -= 0.2
            out = out/0.6
            out = out*test_Data[index][17]

            # Determine if estimate is worthy of investment
            lstObsrv = test_Data[index][1]
            actual = test_Data[index+PP][1]

            # Predicted % Change
            changePerc.append(((out-lstObsrv)/lstObsrv)*100)
            # Actual % change
            aChange.append(((actual-lstObsrv)/lstObsrv)*100)

            outputs.append(out)
            targets.append(actual)
            inputs.append(lstObsrv)
            if ((out - (math.fabs(upperBound)*out)) > lstObsrv):
                nInvMade += 1
                invMade += lstObsrv
                returnMade += actual
                if (lstObsrv < actual):
                    nInvCorr += 1
            # Calculations for evaluation statistics
            if (lstObsrv < out):
                # Ascending estimate
                nAsc += 1
                if (lstObsrv < actual):
                    nCorr += 1
                    nAscCorr += 1
            else:
                # Descending estimate
                nDesc += 1
                if (lstObsrv > out):
                    nCorr += 1

        # Calculate Evaluation statistics
        AscCorr = (float(nAscCorr)/float(nAsc))*100
        CorrRatio = (float(nCorr)/(float(nAsc)+float(nDesc)))*100
        print("Correct %: " + str(CorrRatio))
        print("Profitable Correct: " + str(AscCorr))
        if nInvMade > 0:
            ROI = returnMade/invMade
            print("Number of investments: " + str(nInvMade))
            print("Number of investments correct: " + str(nInvCorr))
            print("Investment: " + str(invMade))
            print("Return: " + str(returnMade))
            print("ROI: " + str(ROI))
        else:
            print("No investments could be made.")

        #tPlot, = plt.plot(np.arange(len(targets)), targets)
        cPlot, = plt.plot(np.arange(len(changePerc)), changePerc)
        aPlot, = plt.plot(np.arange(len(aChange)), aChange)
        iPlot, = plt.plot(np.arange(len(inputs)), inputs)
        oPlot, = plt.plot(np.arange(len(outputs)), outputs)
        plt.legend([iPlot, oPlot, cPlot, aPlot], ["Last Observed Price", "Predicted Price", "predicted change %", "actual change %"], loc=1)
        plt.xlabel("Test Run")
        #plt.show()

def getData(filename):
    readfile = open(filename, 'r')
    test_Data = []
    for line in readfile:
        line = line.strip()
        line = line.split("\t")
        for x in range(1,18):
            line[x] = float(line[x])
        line[16] = line[16]/100
        test_Data.append(line)
    readfile.close()
    return test_Data


def actFunc(x):
    return 1 / ( 1 + math.exp(-x) )

def dActFunc(x):
    return x * (1 - x)

def sigmoidMatrix(matrix):
    rows = matrix.shape[0]
    if len(matrix.shape) > 1:
        cols = matrix.shape[1]
    else:
        cols = 1
    for x in range(0, rows):
        for y in range(0, cols):
            matrix[x, y] = actFunc(matrix[x, y])
    return matrix

def derivativeSigmoid(matrix):
    rows = matrix.shape[0]
    if len(matrix.shape) > 1:
        cols = matrix.shape[1]
    else:
        cols = 1
    for x in range(0, rows):
        for y in range(0, cols):
            matrix[x, y] = dActFunc(matrix[x, y])
    return matrix

def main():
    # Initialise MLP with 15 inputs, h Hidden Nodes, and 1 Output.
    h = 30
    mlp = MultiLayerPerceptron(15, h, 1);
    # Prediction Period
    PP = 7
    # filename
    filename = "NormINTC.txt"
    # Z-Value
    zValue = 1.15

    data = getData(filename)

    start = datetime.now()
    upperBound = mlp.beginTraining(PP, data[:1934], zValue)


    mlp.testModel(PP, data[1934:], upperBound)
    print(datetime.now() - start)

    return mlp

MLP = main()
