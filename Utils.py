import numpy as np 
import Evaluation
import random


def getProjectPath():
	return './'



def runningAverage(inputData, width):
    inputData = np.atleast_2d(inputData)
    target = np.zeros((inputData.shape))
    for i in range(width,len(inputData-width)):
            target[i,:] = np.mean(inputData[i-width:i+width,:],0)
    return target

def writeToReportFile(text):
    print(getProjectPath()+'results/report.csv')
    with open(getProjectPath()+'results/report.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(text)

def splitBySignals(dataStep):
    segments= []
    for input, target in dataStep:
        targetInt = np.argmax(Evaluation.addNoGestureSignal(target), 1)
        inds= np.where(targetInt[:-1]!= targetInt[1:])[0]
        lastInd = -1
        for ind in inds:
            if targetInt[ind] != np.max(targetInt):
                iSegment = input[lastInd+1:ind+1]
                tSegement = target[lastInd+1:ind+1]
                tSegement[0,:]=0
                tSegement[-1,:]=0
                segments.append((iSegment,tSegement))
                lastInd = ind
        ind = len(targetInt)-1
        iSegment = input[lastInd+1:ind+1]
        tSegement = target[lastInd+1:ind+1]
        tSegement[0,:]=0
        tSegement[-1,:]=0
        segments.append((iSegment,tSegement))
    return segments

def shuffleDataStep(dataStep, nFolds, nRepeat=1):
    segs = splitBySignals(dataStep)
    segs = segs * nRepeat
    random.shuffle(segs)
    segs = [ segs[i::nFolds] for i in range(nFolds) ]
    dataStep=[]
    for segList in segs:
        ind = np.concatenate([x[0] for x in segList],0)
        t   = np.concatenate([x[1] for x in segList],0)
        dataStep.append((ind,t))
    return dataStep
