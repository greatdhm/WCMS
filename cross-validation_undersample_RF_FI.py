import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")

def getCurrentDateTime():
    strCurrentDateTime = time.strftime('%Y.%m.%d_%H.%M.%S')
    return strCurrentDateTime

def computeAcc(act, pred, measure, dc=False):
    cutoffs = np.linspace(0, 1, 11)
    smeasure = sum(measure)
    sorted_pred = sorted(enumerate(pred), key=lambda x: x[1], reverse=not dc)
    index = [i[0] for i in sorted_pred]
    cmeasure = np.cumsum([measure[i] for i in index])
    sbug = sum(act)
    cbug = np.cumsum([act[i] for i in index])

    res = []
    for k in range(len(cutoffs)):
        tindex, max_diff = 0, 100000
        for i in range(len(cmeasure)):
            effort = cmeasure[i] / smeasure
            diff = abs(effort - cutoffs[k])
            if max_diff >= diff:
                max_diff = diff
                tindex = i
        res.append(cbug[tindex] / sbug)

    return res

def computePopt(act, pred, loc, dc=False):
    sortedId = np.argsort(pred)[::-1]

    # cumulative summing
    cloc = np.cumsum([loc[i] for i in sortedId])
    cindp = np.cumsum([act[i] for i in sortedId])

    optId_res = list(enumerate(map(lambda x: x[0] / x[1], zip(act, loc))))
    optId_res = sorted(optId_res, key=lambda x: x[1], reverse=True)
    optId_index = [i[0] for i in optId_res]
    optcloc = np.cumsum([loc[i] for i in optId_index])
    optcindp = np.cumsum([act[i] for i in optId_index])

    minId_res = list(enumerate(map(lambda x: x[0] / x[1], zip(act, loc))))
    minId_res = sorted(minId_res, key=lambda x: x[1], reverse=False)
    minId_index = [i[0] for i in minId_res]
    mincloc = np.cumsum([loc[i] for i in minId_index])
    mincindp = np.cumsum([act[i] for i in minId_index])

    actauc = auc(cloc, cindp)
    optauc = auc(optcloc, optcindp)
    minauc = auc(mincloc, mincindp)

    popt = (actauc - minauc) / (optauc - minauc)

    return popt

# define project name
projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
datapath = r'./data/'
dfResult = pd.DataFrame(columns=['project', 'Regressor', 'acc1', 'popt1', 'RunTime'])
for i in projects:
    start = time.time()
    datafile = datapath + i + '.csv'
    # read data
    data = pd.read_csv(datafile)
    # compute effort
    # effort = LA + LD
    data['codechurn'] = (data['la'] + data['ld']) * data['lt']
    clean_data = data[data['codechurn'] > 0]
    clean_data = clean_data.reset_index(drop=True)
    clean_data['nf_nl'] = (clean_data['nf'] - clean_data['nf'].min(axis=0)) / (clean_data['nf'].max(axis=0) - clean_data['nf'].min(axis=0))
    coe_dif = clean_data['nf_nl'] ** clean_data['entropy']
    clean_data['effort'] = coe_dif * clean_data['codechurn']
    y = clean_data['bug'] / clean_data['effort']
    y = np.asarray(y).ravel()
    X_df = clean_data[['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
    X = X_df.values
    rfc = RandomForestRegressor(n_estimators=100)
    kf = KFold(n_splits=10)
    roi_total = []
    popt_total = []
    rus = RandomUnderSampler()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_us = clean_data['bug'][train_index]
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train_us)
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_df.columns)
        y_resampled_df = pd.DataFrame(y_resampled, columns=['bug'])
        X_resampled_df['codechurn'] = (X_resampled_df['la'] + X_resampled_df['ld']) * X_resampled_df['lt']
        X_resampled_df['nf_nl'] = (X_resampled_df['nf'] - X_resampled_df['nf'].min(axis=0)) / (
                X_resampled_df['nf'].max(axis=0) - X_resampled_df['nf'].min(axis=0))
        coe_dif_os = X_resampled_df['nf_nl'] ** X_resampled_df['entropy']
        X_resampled_df['effort'] = coe_dif_os * X_resampled_df['codechurn']
        y_resampled_dd = y_resampled_df['bug'] / X_resampled_df['effort']
        y_resampled_dd = np.asarray(y_resampled_dd).ravel()
        rfc.fit(X_resampled, y_resampled_dd)
        y_new = rfc.predict(X_test)
        y_act = clean_data['bug'][test_index]
        y_act = y_act.reset_index(drop=True)
        measure = clean_data['effort'][test_index]
        measure = measure.reset_index(drop=True)
        roi = computeAcc(y_act, y_new, measure)
        roi_total.append(roi[1])
        popt = computePopt(y_act, y_new, measure)
        popt_total.append(popt)

    scores = cross_val_score(rfc, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-scores)

    mse1 = rmse_scores.mean()
    acc1 = sum(roi_total) / len(roi_total)
    popt1 = sum(popt_total) / len(popt_total)

    regressor = 'RF'
    runtime = time.time() - start
    s = pd.Series([i, regressor, acc1, popt1, runtime], index=dfResult.columns)
    dfResult = dfResult.append(s, ignore_index=True)

# generate excel file
filename = './result/FirstPaper_CrossValidation_RF_FI_' + getCurrentDateTime() + '.xls'
dfResult.to_excel(filename, index=False)
