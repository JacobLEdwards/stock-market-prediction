def getData(filename):
    readfile = open(filename, 'r')
    # Read Title Line
    readfile.readline()

    # Parse all data into an array of Dates and Floats.
    fileData = []
    for line in readfile:
        line = line.strip()
        line = line.split("\t")
        line[1] = float(line[1])
        try:
            line[6] = float(line[6])
            fileData.append(line[:2] + [line[6]])
        except:
            fileData.append(line[:2])
    readfile.close()

    return fileData

# Write data to new file
def writeData(fileData, filename):
    for x in range(len(fileData)):
        for y in range(1, 17):
            fileData[x][y] = str(fileData[x][y])
        fileData[x] = "\t".join(fileData[x])
    fileData = "\n".join(fileData)
    writefile = open(filename, 'w')
    writefile.write(fileData)
    writefile.close()


# Classify data based on values of x and x+PP
def classify(val1, val2):
    # Classes are:
    # 0 - Rapidly Ascending,    1 - Ascending,          2 - Neutral
    # 3 - Descending,           4 - Rapidly Descending
    diff = (val2 - val1) / val1
    if (diff > 0.15):
        dataClass = "0"
    elif (diff > 0.01):
        dataClass = "1"
    elif (diff > -0.01):
        dataClass = "2"
    elif (diff > -0.15):
        dataClass = "3"
    else:
        dataClass = "4"
    return dataClass

# Label data points in file for various values of PP
def label(filename):
    # for each value of PP
    PPVals = [1, 7, 30]
    for PP in PPVals:
        # get file data
        fileData = getData(filename)
        newFileData = []
        classCount = [0, 0, 0, 0, 0]
        # classify each data point
        for index in range(14, len(fileData)-PP):
            # Get previous 13 close prices
            prevPrices = [fileData[index+x][1] for x in range(-13,0)]
            # Classify data point
            val1 = fileData[index][1]
            val2 = fileData[index+PP][1]
            dataClass = classify(val1, val2)
            classCount[int(dataClass)] += 1
            newFileData.append([fileData[index][0], fileData[index][1]] + prevPrices + [val1, fileData[index][2], dataClass])
            # Normalise data
            newIndex = len(newFileData)-1
            maxVal = max(newFileData[newIndex][2:16])
            for x in range(2, 16):
                newFileData[newIndex][x] = newFileData[newIndex][x]/maxVal
        # write new file
        newFileName = str(PP) + filename
        writeData(newFileData, newFileName)
        # output data
        print newFileName
        print classCount

fileArray = ["AMD.txt", "BA.txt", "INTC.txt", "KO.txt", "TSCO.L.txt"]
for file in fileArray:
    label(file)
