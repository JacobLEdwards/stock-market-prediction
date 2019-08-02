import math

def getData(filename, trainingRun):
    readfile = open(filename, 'r')
    # Remove Title Line
    readfile.readline()

    # Parse all data into an array of Dates and Floats.
    testData = []
    for line in readfile:
        line = line.strip()
        line = line.split("\t")
        for x in range(1, len(line)):
            line[x] = float(line[x])
        testData.append(line)
    readfile.close()

    # Discard data put aside for training other methods.
    if (trainingRun):
        testData = testData[:1934]
    else:
        testData = testData[1934:]
    return testData

def calculateCoefficients(N, PP, index, calcData):
    # Calculation Values
    xBar, yBar, sXY, sXsq = 0, 0, 0, 0

    # Calculate Precursor Values
    for i in range(1,N+1):
        xBar += i
        yBar += calcData[i][1]
        sXY += i*float(calcData[i][1])
        sXsq += i**2

    # Calculate xBar, yBar
    xBar = float(xBar) / N
    yBar = float(yBar) / N

    #Calculate A and B coefficients for OLS.
    B = (sXY - (N*xBar*yBar))/(sXsq - N*(xBar**2))
    A = yBar - (B*xBar)
    return A, B

# Use OLS coefficients to estimate the value of the stock at X = index+N+PP
def estimate(N, PP, A, B, index, calcData):
    xData = index+N+PP
    estimValue = (A + (B*(N+PP)))
    actualValue = calcData[N+PP][1]
    errValue = math.fabs(estimValue - actualValue)/actualValue
    # returns index, estimate, actual value, and % error between estimate and actual.
    return xData, estimValue, actualValue, errValue

testData = []
def main():
    # Number of Predictors
    N = 7
    # Prediction Period
    PP = 30
    # z-value for confidence threshold
    zValue = 1.15
    # Stock to be tested
    filename = "KO.txt"

    print filename
    print "N = " + str(N)
    print "PP = " + str(PP)
    print "Z = " + str(zValue) + "\n"

    # Array of indices, estimates, actual values, and error values
    indArray, estArray, actArray, errArray = [], [], [], []

    # -- Training Run --
    testData = getData(filename, True)
    # Test for each set of data in the valid range.
    sError = 0
    nTests = len(testData)-(N+PP)
    for index in range(nTests):
        # Data used for current test.
        calcData = testData[index:index+N+PP+1]
        A, B = calculateCoefficients(N, PP, index, calcData)
        ret = estimate(N, PP, A, B, index, calcData)
        indArray.append(ret[0])
        estArray.append(ret[1])
        actArray.append(ret[2])
        errArray.append(ret[3])
        sError += ret[3]
    # Calculate Data for Threshold Algorithm
    avgError = sError/nTests
    sErrorSquared = 0
    for error in errArray:
        sErrorSquared += (error - avgError)**2
    stdDev = math.sqrt(sErrorSquared / (nTests-1))
    upperBound = avgError + zValue * stdDev
    print "avgError: " + str(avgError)
    print "stdDev: " + str(stdDev)
    print "Upper Error Bound: " + str(upperBound)

    # -- Testing Run --
    indArray, estArray, actArray, errArray = [], [], [], []
    nInvMade = 0
    invMade = 0
    returnMade = 0
    nAsc = 0
    nDesc = 0
    nCorr = 0
    nAscCorr = 0
    testData = getData(filename, False)
    nTests = len(testData)-(N+PP)
    # Test for each set of data in the valid range.
    for index in range(nTests):
        # Data used for current test.
        calcData = testData[index:index+N+PP+1]
        A, B = calculateCoefficients(N, PP, index, calcData)
        ret = estimate(N, PP, A, B, index, calcData)
        indArray.append(ret[0])
        estArray.append(ret[1])
        actArray.append(ret[2])
        errArray.append(ret[3])
        # Determine if estimate is worthy of investment
        lstObsrv = testData[index+N-1][1]
        if ((ret[1] - (math.fabs(upperBound)*ret[1])) > lstObsrv):
            nInvMade += 1
            invMade += lstObsrv
            returnMade += ret[2]
        # Calculations for evaluation statistics
        if (lstObsrv < ret[1]):
            # Ascending estimate
            nAsc += 1
            if (lstObsrv < ret[2]):
                nCorr += 1
                nAscCorr += 1
        else:
            # Descending estimate
            nDesc += 1
            if (lstObsrv > ret[2]):
                nCorr += 1
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

main()
