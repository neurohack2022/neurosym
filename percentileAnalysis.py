import numpy as np

csvDataFocus = open("test-focus")
csvDataRelax = open("test-relax")
focusArray2D = np.genfromtxt(csvDataFocus, delimiter=",")
relaxArray2D = np.genfromtxt(csvDataRelax, delimiter=",")

focusCounter = 0

focusAvg = []

for channelData in focusArray2D:
    indexList = [np.any(i) for i in np.isnan(channelData)]
    channelDataClean = np.delete(channelData, indexList, axis=0)
    # print(channelDataClean)
    
    focusAvg.append(sum(channelDataClean)/len(channelData))


totalLength = len(focusAvg)
print(totalLength)

for ratio in focusAvg:
    if ratio > 0.7:
        focusCounter += 1
    
print(focusCounter/totalLength)


relaxCounter = 0
relaxAvg = []

print(relaxArray2D)
for channelData in relaxArray2D:
    indexList = [np.any(i) for i in np.isnan(channelData)]
    channelDataClean = np.delete(channelData, indexList, axis=0)
    # print(channelDataClean)
    
    relaxAvg.append(sum(channelDataClean)/len(channelData))

totalLength = len(relaxAvg)
print(totalLength)

for ratio in focusAvg:
    if ratio < 0.7 and ratio > 0:
        relaxCounter += 1
    
print(relaxCounter/totalLength)

    