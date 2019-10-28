import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import Main

from DataSet import createData
from Evaluation import calcMaxActivityPrediction, calcInputSegmentSeries, postProcessPrediction, calcTPFPForThresholds, calcLevenshteinError, addTresholdSignal, plot_confusion_matrix

def evaluateTestFile(iFile,inputGestures,usedGestures, gestureNames, totalGestureNames, reservoir, bestFlow, tresholds, bestF1ScoreTreshold, shuffle, learnTreshold, f1Scores,f1BestPossibleScores, f1ppScores, f1maxAppScores, f1maxAppBestPossibleScores, f1ScoreNames, accuracies, levs, levs_pp, pp, confMatrices):
    testData = createData(iFile, inputGestures, usedGestures)
    if shuffle:
        testData = Main.shuffleDataStep([testData], 1)[0]
     

    t_target = testData[1]
    t_prediction = bestFlow(testData[0])
    if learnTreshold:
        learnedTreshold = t_prediction[:,-1]
        t_prediction = t_prediction[:,:-1]
        tresholds = tresholds[:-1]
        t_maxApp_prediction = calcMaxActivityPrediction(t_prediction,t_target,learnedTreshold,10)
    else:    
        learnedTreshold = np.ones((t_prediction.shape[1],1))*bestF1ScoreTreshold
        t_maxApp_prediction = calcMaxActivityPrediction(t_prediction,t_target,bestF1ScoreTreshold,10)
        #t_maxApp_prediction = calcMaxActivityPrediction(t_prediction,t_target,0.4,1)
    #t_prediction = t_maxApp_prediction
    t_pp_prediction = postProcessPrediction(t_prediction, tresholds)

    _, bestPossibleMaxAppF1Score,_ =  calcTPFPForThresholds(t_prediction, t_target, iFile+' - target treshold', False)
    pp.savefig()
    _, bestPossibleF1Score,_ =  calcTPFPForThresholds(t_prediction, t_target, iFile+' - max target', True)
    pp.savefig()
    
    lev = calcLevenshteinError(t_prediction, t_target, 0.4)
    lev_pp = calcLevenshteinError(t_maxApp_prediction, t_target, 0.5)
    levs.append(lev)
    levs_pp.append(lev_pp)
    fig = plt.figure(figsize=(30,30))
    fig.suptitle(iFile)
    plt.clf()
    ax1 = plt.subplot(211)
    plt.title('Prediction on test ' +iFile)
    cmap = mpl.cm.jet
    for i in range(t_prediction.shape[1]):
        plt.plot(t_prediction[:,i],c=cmap(float(i)/t_prediction.shape[1]),label=totalGestureNames[usedGestures[i]],linewidth=3)
        plt.fill_between(range(len(t_prediction)), np.max(t_prediction,1), np.squeeze(np.ones((len(t_prediction),1))), where=testData[1][:,i]==1,facecolor=cmap(float(i)/t_prediction.shape[1]), alpha=0.3)
        #plt.fill_between(range(len(t_prediction)), 1.4, 1.6, where=testData[1][:,i]==1,facecolor=cmap(float(i)/t_prediction.shape[1]), alpha=0.7)
        #plt.fill_between(range(len(t_prediction)), 1.2, 1.4, where=t_prediction[:,i]==np.max(addTresholdSignal(t_prediction,0.4),1),facecolor=cmap(float(i)/t_prediction.shape[1]), alpha=0.7)
        #plt.fill_between(range(len(t_prediction)), 1.0, 1.2, where=t_maxApp_prediction[:,i]==1,facecolor=cmap(float(i)/t_prediction.shape[1]), alpha=0.7)    
        plt.fill_between(range(len(t_prediction)), 0, t_prediction[:,i], where=t_prediction[:,i]==np.max(t_prediction,1), facecolor=cmap(float(i)/t_prediction.shape[1]), alpha=0.5)
    if learnTreshold:
        plt.plot(learnedTreshold,c='black',label='treshold',linewidth=3)
        
    for limCounter in range(5):
        plt.annotate('Target', xy=(limCounter*1000+1,1.45))
        plt.annotate('ArgMax Prediction',xy=(limCounter*1000+1,1.25))
        plt.annotate('Segmented Prediction',xy=(limCounter*1000+1,1.05))
    plt.legend()
    plt.subplot(212, sharex=ax1)
    plt.title('Input')
    if(bestFlow[0].useNormalized==1):
        plt.plot(testData[0][:,0:3]/reservoir.colStdFactor[0:3],label='Fused')
        plt.plot(testData[0][:,3:6]/reservoir.colStdFactor[3:6],label='Rot')
        plt.plot(testData[0][:,6:9]/reservoir.colStdFactor[6:9],label='Acc')
    elif (bestFlow[0].useNormalized==2):
        plt.plot(testData[0][:,0:3]/reservoir.colMaxFactor[0:3],label='Fused')
        plt.plot(testData[0][:,3:6]/reservoir.colMaxFactor[3:6],label='Rot')
        plt.plot(testData[0][:,6:9]/reservoir.colMaxFactor[6:9],label='Acc')
    plt.plot(np.sum(np.abs(bestFlow[0].states),1)/100,label='Res Energy /100')
    #plt.plot(testData[1])
    plt.legend()
    plt.xlabel('timestep')
    
    for limCounter in range(5):
        plt.xlim(limCounter*1000,(limCounter+1)*1000)
        pp.savefig()
     
    pred, targ = calcInputSegmentSeries(t_prediction, t_target, 0.4, False)
    pp_pred, pp_targ = calcInputSegmentSeries(t_pp_prediction, t_target, 0.05, False)
    pred_maxApp, targ_maxApp = calcInputSegmentSeries(t_maxApp_prediction, t_target, 0.5)
    
    cm = sklearn.metrics.confusion_matrix(targ, pred)
    confMatrices.append(cm)
    fig1 = plot_confusion_matrix(cm,gestureNames,iFile)
    pp.savefig()
     
    pp_cm = sklearn.metrics.confusion_matrix(pp_targ, pp_pred)
    fig2 = plot_confusion_matrix(pp_cm,gestureNames,'pp_'+iFile)
    pp.savefig()
    
    maxApp_cm = sklearn.metrics.confusion_matrix(targ_maxApp, pred_maxApp)
    plot_confusion_matrix(maxApp_cm,gestureNames,'maxApp_'+iFile)
    pp.savefig()
    accuracies.append(sklearn.metrics.accuracy_score(targ_maxApp,pred_maxApp))
    
    
    f1 = np.mean(sklearn.metrics.f1_score(targ,pred,average=None))
    f1_pp = np.mean(sklearn.metrics.f1_score(pp_targ,pp_pred,average=None))
    f1_maxApp = np.mean(sklearn.metrics.f1_score(targ_maxApp,pred_maxApp,average=None))
        
    f1Scores.append(f1)
    f1ppScores.append(f1_pp)
    f1maxAppBestPossibleScores.append(bestPossibleMaxAppF1Score)
    f1BestPossibleScores.append(bestPossibleF1Score)
    f1ScoreNames.append(iFile)    
    f1maxAppScores.append(f1_maxApp)
    
    return t_target,t_prediction, t_pp_prediction, t_maxApp_prediction, learnedTreshold
