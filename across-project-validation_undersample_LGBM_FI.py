import math
import numpy as np
import pandas as pd
import time
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMRegressor
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
    sortedId = np.argsort(pred)[::-1]  # 降序排列

    # cumulative summing
    cloc = np.cumsum([loc[i] for i in sortedId])
    cindp = np.cumsum([act[i] for i in sortedId])

    # compute optimal model
    optId_res = list(enumerate(map(lambda x: x[0] / x[1], zip(act, loc))))
    optId_res = sorted(optId_res, key=lambda x: x[1], reverse=True)
    optId_index = [i[0] for i in optId_res]
    optcloc = np.cumsum([loc[i] for i in optId_index])
    optcindp = np.cumsum([act[i] for i in optId_index])

    # 升序排列，变更工作量大的会排在前面
    minId_res = list(enumerate(map(lambda x: x[0] / x[1], zip(act, loc))))
    minId_res = sorted(minId_res, key=lambda x: x[1], reverse=False)
    minId_index = [i[0] for i in minId_res]
    mincloc = np.cumsum([loc[i] for i in minId_index])
    mincindp = np.cumsum([act[i] for i in minId_index])

    # compute AUC of the three plot
    actauc = auc(cloc, cindp)
    optauc = auc(optcloc, optcindp)
    minauc = auc(mincloc, mincindp)

    # opt = calcOPT2(optauc, minauc, auc)
    popt = (actauc - minauc) / (optauc - minauc)

    return popt


# define project name
train_projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
test_projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
datapath = r'./data/'
dfResult = pd.DataFrame(columns=['project1', 'project2', 'Regressor', 'acc1', 'popt1', 'RunTime'])
for i1 in train_projects:
    train_datafile = datapath + i1 + '.csv'
    # read data
    train_data = pd.read_csv(train_datafile)
    # compute effort
    # effort = LA + LD
    train_data['codechurn'] = (train_data['la'] + train_data['ld']) * train_data['lt']

    train_clean_data = train_data[train_data['codechurn'] > 0]
    train_clean_data = train_clean_data.reset_index(drop=True)
    LGBM = LGBMRegressor()
    roi_total = []
    # Popt
    popt_total = []
    rus = RandomUnderSampler()
    X_train_df = train_clean_data[
        ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
    X_train = X_train_df.values
    y_train = train_clean_data['bug']
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_train_df.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=['bug'])
    X_resampled_df['codechurn'] = (X_resampled_df['la'] + X_resampled_df['ld']) * X_resampled_df['lt']
    X_resampled_df['nf_nl'] = (X_resampled_df['nf'] - X_resampled_df['nf'].min(axis=0)) / (
            X_resampled_df['nf'].max(axis=0) - X_resampled_df['nf'].min(axis=0))
    train_coe_dif = X_resampled_df['nf_nl'] ** X_resampled_df['entropy']
    X_resampled_df['effort'] = train_coe_dif * X_resampled_df['codechurn']
    y_resampled_dd = y_resampled_df['bug'] / X_resampled_df['effort']
    y_resampled_dd = np.asarray(y_resampled_dd).ravel()

    LGBM.fit(X_resampled, y_resampled_dd)

    for i2 in test_projects:
        start = time.time()  # start time
        if i1 == i2:
            continue
        else:
            test_datafile = datapath + i2 + '.csv'
            # read data
            test_data = pd.read_csv(test_datafile)
            # compute effort
            # effort = LA + LD
            test_data['codechurn'] = (test_data['la'] + test_data['ld']) * test_data['lt']
            # 删除codechurn = 0的列
            test_clean_data = test_data[test_data['codechurn'] > 0]
            # 重置clean_data的索引
            test_clean_data = test_clean_data.reset_index(drop=True)

            test_clean_data['nf_nl'] = (test_clean_data['nf'] - test_clean_data['nf'].min(axis=0)) / (
                    test_clean_data['nf'].max(axis=0) - test_clean_data['nf'].min(axis=0))
            # 设计codechurn的难度系数coefficient_difficulty,简写为coe_dif
            test_coe_dif = test_clean_data['nf_nl'] ** test_clean_data['entropy']
            # 工作量 = 难度系数 * code churn
            test_clean_data['effort'] = test_coe_dif * test_clean_data['codechurn']

            X_test_df = test_clean_data[
                ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
            X_test = X_test_df.values

            y_new = LGBM.predict(X_test)
            y_act = test_clean_data['bug']
            measure = test_clean_data['effort']
            roi = computeAcc(y_act, y_new, measure)
            popt = computePopt(y_act, y_new, measure)
            print('Train project:%s, Test project:%s, Acc: %s, Popt: %s' % (i1, i2, roi[1], popt))

            regressor = 'LGBM'
            runtime = time.time() - start
            s = pd.Series([i1, i2, regressor, roi[1], popt, runtime], index=dfResult.columns)
            dfResult = dfResult.append(s, ignore_index=True)


# generate excel file
filename = './result/FirstPaper_AcrossProject_Validation_LGBM_FI_' + getCurrentDateTime() + '.xls'
dfResult.to_excel(filename, index=False)
