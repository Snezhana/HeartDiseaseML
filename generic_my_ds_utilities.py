import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from my_ds_utilities import *
from woe import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectPercentile

def getStatsForFalsTrue(X, y, y_pred):
    fp = X.loc[np.logical_and(y != y_pred, y == False)]
    tp = X.loc[np.logical_and(y == y_pred, y == True)]
    tn = X.loc[np.logical_and(y == y_pred, y == False)]
    fn = X.loc[np.logical_and(y != y_pred, y == True)]
    colStatsTr = {}
    for col in X_test.columns:
        fn_tp_mean = abs(fn[col].mean() - tp[col].mean())
        fn_tp_med = abs(fn[col].median() - tp[col].median())
        tn_fp_mean = abs(tn[col].mean() - fp[col].mean())
        tn_fp_med = abs(tn[col].median() - fp[col].median())
        tn_fn_mean = abs(tn[col].mean() - fn[col].mean())
        tn_fn_med = abs(tn[col].median() - fn[col].median())
        fp_tp_mean = abs(fp[col].mean() - tp[col].mean())
        fp_tp_med = abs(fp[col].median() - tp[col].median())
        allstat = [fn_tp_mean, fn_tp_med, tn_fp_mean, tn_fp_med, tn_fn_mean, tn_fn_med, fp_tp_mean, fp_tp_med]
        colStatsTr[col]  = allstat
    colStatsTr = pd.DataFrame(colStatsTr).transpose()
    colStatsTr.columns =['fn_tp_mean', 'fn_tp_med', 'tn_fp_mean', 'tn_fp_med', 'tn_fn_mean', 'tn_fn_med', 'fp_tp_mean', 'fp_tp_med']
    colStatsTr['fn_tp_mean_tn_fn_mean']= colStatsTr.fn_tp_mean - colStatsTr.tn_fn_mean
    colStatsTr['tn_fp_mean_fp_tp_mean']= colStatsTr.tn_fp_mean - colStatsTr.fp_tp_mean
    colStatsTr['fn_tp_med_tn_fn_med']= colStatsTr.fn_tp_med - colStatsTr.tn_fn_med
    colStatsTr['tn_fp_med_fp_tp_med']= colStatsTr.tn_fp_med - colStatsTr.fp_tp_med
    colStatsTr.index = predictorsCols
    colStatsTr.sort_values('fn_tp_mean_tn_fn_mean')
    return colStatsTr

def getMetricsForPreictions(clfs, X, y):
    X_train, X_test,y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    accuracyCV1 = []
    accuracyCV2 = []
    accuracyCV3 = []
    accuracyCV4 = []
    accuracyCV5 = []
    accuracyCVAvg = []
    precisionTrain0 = []
    precisionTrain1 = []
    precisionTest0 = []
    precisionTest1 = []
    recallTrain0 = []
    recallTrain1 = []
    recallTest0 = []
    recallTest1 = []
    f1Train0 = []
    f1Train1 = []
    f1Test0 = []
    f1Test1 = []
    logLossTrain = []
    logLossTest = []
    tnTest =[]
    tnTrain =[]
    fnTest = []
    fnTrain = []
    fpTest = []
    fpTrain = []
    tpTest = []
    tpTrain = []
    accTest = []
    accTrain = []
    for clf in clfs:
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        y_test_predp = clf.predict_proba(X_test)
        y_train_predp = clf.predict_proba(X_train)
        precisionTrain0.append(precision_score(y_train, y_pred_train, average=None)[0])
        precisionTrain1.append(precision_score(y_train, y_pred_train, average=None)[1])
        precisionTest0.append(precision_score(y_test, y_pred_test, average=None)[0])
        precisionTest1.append(precision_score(y_test, y_pred_test, average=None)[1])
        recallTrain0.append(recall_score(y_train, y_pred_train, average=None)[0])
        recallTrain1.append(recall_score(y_train, y_pred_train, average=None)[1])
        recallTest0.append(recall_score(y_test, y_pred_test, average=None)[0])
        recallTest1.append(recall_score(y_test, y_pred_test, average=None)[1])
        f1Train0.append(f1_score(y_train, y_pred_train, average=None)[0])
        f1Train1.append(f1_score(y_train, y_pred_train, average=None)[1])
        f1Test0.append(f1_score(y_test, y_pred_test, average=None)[0])
        f1Test1.append(f1_score(y_test, y_pred_test, average=None)[1])
        logLossTrain.append(log_loss(y_train, y_train_predp))
        logLossTest.append(log_loss(y_test, y_test_predp))
        tnTrain.append(confusion_matrix(y_train, y_pred_train)[0,0])
        tnTest.append(confusion_matrix(y_test, y_pred_test)[0,0])
        fnTrain.append(confusion_matrix(y_train, y_pred_train)[0,1])
        fnTest.append(confusion_matrix(y_test, y_pred_test)[0,1])
        fpTrain.append(confusion_matrix(y_train, y_pred_train)[1,0])
        fpTest.append(confusion_matrix(y_test, y_pred_test)[1,0])
        tpTrain.append(confusion_matrix(y_train, y_pred_train)[1,1])
        tpTest.append(confusion_matrix(y_test, y_pred_test)[1,1])
        cvScores = cross_val_score(clf, X, y, cv=5)
        cvScores.sort()
        accuracyCV1.append(cvScores[0])
        accuracyCV2.append(cvScores[1])
        accuracyCV3.append(cvScores[2])
        accuracyCV4.append(cvScores[3])
        accuracyCV5.append(cvScores[4])
        accuracyCVAvg.append(cvScores.mean())
        accTest.append(clf.score(X_test, y_test))
        accTrain.append(clf.score(X_train, y_train))
        
        
    allMetrics = [ precisionTrain0, precisionTrain1, precisionTest0,precisionTest1, recallTrain0,
                  recallTrain1, recallTest0, recallTest1, f1Train0, f1Train1, f1Test0, f1Test1, logLossTrain, logLossTest,
                  tnTest,tnTrain, fnTest, fnTrain, fpTest, fpTrain, tpTest, tpTrain, accuracyCV1, accuracyCV2, accuracyCV3, 
                  accuracyCV4, accuracyCV5, accuracyCVAvg, accTest, accTrain]
    dfmet = pd.DataFrame(allMetrics).T
    dfmet.columns = [ 'precisionTrain0', 'precisionTrain1', 'precisionTest0','precisionTest1', 'recallTrain0',
                  'recallTrain1', 'recallTest0', 'recallTest1', 'f1Train0', 'f1Train1', 'f1Test0', 'f1Test1', 'logLossTrain', 'logLossTest',
                  'tnTest','tnTrain', 'fpTest', 'fpTrain', 'fnTest', 'fnTrain', 'tpTest', 'tpTrain', 'accuracyCV1', 'accuracyCV2', 'accuracyCV3', 
                  'accuracyCV4', 'accuracyCV5', 'accuracyCVAvg', 'accTest', 'accTrain']
    return dfmet

def logLoss(df, true, pred):
    logloss = -(df[true] * np.log(df[pred]) + ((1-df[true]) * np.log(1-df[pred]))).sum()/len(df)
    return logloss


def create_pig_table(basetable, target, variable):
    # Create groups for each variable
    groups = basetable[[target,variable]].groupby(variable)
    
    # Calculate size and target incidence for each group
    pig_table = groups[target].agg({'Incidence' : np.mean, 'Size' : np.size}).reset_index()

    pig_table['percent_size'] = pig_table['Size']/pig_table['Size'].sum()*100
    pig_table['importance'] = abs(0.5 - pig_table['Incidence'])*pig_table['percent_size']
    # Return the predictor insight graph table
    return pig_table

# The function to plot a predictor insight graph
def plot_pig(pig_table, variable):
    
    # Plot formatting
    plt.ylabel("Size", rotation=0, rotation_mode="anchor", ha="right")
    
    # Plot the bars with sizes 
    pig_table["Size"].plot(kind="bar", width=0.5, color="lightgray", edgecolor="none") 
    
    # Plot the incidence line on secondary axis
    pig_table["Incidence"].plot(secondary_y=True)
    
    # Plot formatting
    plt.xticks(np.arange(len(pig_table)), pig_table[variable])
    plt.xlim([-0.5, len(pig_table) - 0.5])
    plt.ylabel("Incidence", rotation=0, rotation_mode="anchor", ha="left")
    
    # Show the graph
    plt.show()

def getWoe(df):
    """
    returns the WoE calulated for all categorical features in the dataframe(df)
    based on 500 times random selection of 160 from 180 samples 
    """
    w = WOE()
    woes = []
    numberOfSamping = 500
    
    for i in range(0,numberOfSamping):
        samp = df.sample(n=160, random_state=i+200)
        X = samp.drop('heart_disease_present', axis=1)
        y =samp['heart_disease_present']
        z, woesdict = w.woe(X,y,replace=True)
        woes.append(woesdict)
    dfwoe = pd.DataFrame(woes)
    totalWoes = {}

    
    for col in dfwoe.columns:
        temp = {}
        for row in dfwoe[col]:
            for a, b in row.items():
                temp[a] = (b)/numberOfSamping + temp.get(a, 0)
        totalWoes[col] = temp
    return totalWoes
