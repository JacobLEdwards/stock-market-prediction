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
        for y in range(1, 18):
            fileData[x][y] = str(fileData[x][y])
        fileData[x] = "\t".join(fileData[x])
    fileData = "\n".join(fileData)
    writefile = open(filename, 'w')
    writefile.write(fileData)
    writefile.close()

# Normalise data points in file
def norm(filename):
    # get file data
    fileData = getData(filename)
    newFileData = []
    classCount = [0, 0, 0, 0, 0]
    # classify each data point
    for index in range(14, len(fileData)):
        # Get previous 13 close prices
        prevPrices = [fileData[index+x][1] for x in range(-13,0)]
        val1 = fileData[index][1]

        newFileData.append([fileData[index][0], fileData[index][1]] + prevPrices + [val1, fileData[index][2]])

        # Normalise data
        newIndex = len(newFileData)-1
        maxVal = max(newFileData[newIndex][2:16])
        for x in range(2, 16):
            val = newFileData[newIndex][x]/maxVal
            val = val*0.6
            val += 0.2
            newFileData[newIndex][x] = val
        newFileData[newIndex].append(maxVal)

    # write new file
    newFileName = "Norm" + filename
    writeData(newFileData, newFileName)

fileArray = ["AMD.txt", "BA.txt", "INTC.txt", "KO.txt", "TSCO.L.txt"]
for file in fileArray:
    norm(file)
