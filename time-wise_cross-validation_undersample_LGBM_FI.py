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

projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
datapath = r'./data/'
dfResult = pd.DataFrame(columns=['project', 'Regressor', 'acc1', 'popt1', 'RunTime'])
for i in projects:
    start = time.time()
    datafile = datapath + i + '.csv'
    data = pd.read_csv(datafile)
    data['codechurn'] = (data['la'] + data['ld']) * data['lt']
    clean_data = data[data['codechurn'] > 0]
    clean_data = clean_data.reset_index(drop=True)
    clean_data['nf_nl'] = (clean_data['nf'] - clean_data['nf'].min(axis=0)) / (
            clean_data['nf'].max(axis=0) - clean_data['nf'].min(axis=0))
    coe_dif = clean_data['nf_nl'] ** clean_data['entropy']
    clean_data['effort'] = coe_dif * clean_data['codechurn']
    clean_data['ym'] = pd.to_datetime(clean_data['commitdate']).dt.strftime('%Y-%m')
    timewise_data = clean_data.sort_values(by='ym')
    timewise_data['group_num'] = timewise_data.groupby('ym').ngroup() + 1
    month_nums = sorted(set(timewise_data['group_num']))
    LGBM = LGBMRegressor()
    roi_total = []
    popt_total = []
    rus = RandomUnderSampler()

    for j in month_nums:
        if j > len(month_nums) - 6:
            break
        train_mask = (timewise_data['group_num'] >= j) & (timewise_data['group_num'] <= j + 5)
        train_data = timewise_data.loc[train_mask]
        test_data = timewise_data[timewise_data['group_num'] == j + 6]
        X_train_df = train_data[
            ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
        X_train = X_train_df.values
        y_train = train_data['bug']
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_train_df.columns)
        y_resampled_df = pd.DataFrame(y_resampled, columns=['bug'])
        X_resampled_df['codechurn'] = (X_resampled_df['la'] + X_resampled_df['ld']) * X_resampled_df['lt']
        X_resampled_df['nf_nl'] = (X_resampled_df['nf'] - X_resampled_df['nf'].min(axis=0)) / (
                X_resampled_df['nf'].max(axis=0) - X_resampled_df['nf'].min(axis=0))
        coe_dif_os = X_resampled_df['nf_nl'] ** X_resampled_df['entropy']
        X_resampled_df['effort'] = coe_dif_os * X_resampled_df['codechurn']
        y_resampled_dd = y_resampled_df['bug'] / X_resampled_df['effort']
        y_resampled_dd = np.asarray(y_resampled_dd).ravel()
        LGBM.fit(X_resampled, y_resampled_dd)
        X_test_df = test_data[
            ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
        X_test = X_test_df.values

        y_new = LGBM.predict(X_test)
        y_act = test_data['bug']
        y_act = y_act.reset_index(drop=True)
        measure = test_data['effort']
        measure = measure.reset_index(drop=True)
        roi = computeAcc(y_act, y_new, measure)
        if math.isnan(roi[1]) or roi[1] == 0:
            continue
        else:
            roi_total.append(roi[1])
        popt = computePopt(y_act, y_new, measure)
        if math.isnan(popt) or popt == 0:
            continue
        else:
            popt_total.append(popt)

    acc1 = sum(roi_total) / len(roi_total)
    popt1 = sum(popt_total) / len(popt_total)

    print('Project name:%s Mean Acc: %s Mean Popt: %s' % (
        i, format(sum(roi_total) / len(roi_total)), format(sum(popt_total) / len(popt_total))))

    regressor = 'LGBM'
    runtime = time.time() - start  # run time
    s = pd.Series([i, regressor, acc1, popt1, runtime], index=dfResult.columns)
    dfResult = dfResult.append(s, ignore_index=True)

# generate excel file
filename = './result/FirstPaper_TimeWise_CrossValidation_LGBM_FI_' + getCurrentDateTime() + '.xls'
dfResult.to_excel(filename, index=False)
