import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import SMA_MO
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
import warnings
from sklearn.model_selection import train_test_split

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
dfResult = pd.DataFrame(columns=['project', 'acc1', 'popt1', 'acc2', 'popt2', 'tuningtime'])
p1 = 0
for i in projects:
    datafile = datapath + i + '.csv'
    data = pd.read_csv(datafile)
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
    LGBM = LGBMRegressor(n_estimators=100)
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
        LGBM.fit(X_resampled, y_resampled_dd)
        y_new = LGBM.predict(X_test)
        y_act = clean_data['bug'][test_index]
        y_act = y_act.reset_index(drop=True)
        measure = clean_data['effort'][test_index]
        measure = measure.reset_index(drop=True)
        roi = computeAcc(y_act, y_new, measure)
        roi_total.append(roi[1])
        popt = computePopt(y_act, y_new, measure)
        popt_total.append(popt)

    scores = cross_val_score(LGBM, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-scores)

    mse1 = rmse_scores.mean()
    acc1 = sum(roi_total) / len(roi_total)
    popt1 = sum(popt_total) / len(popt_total)

    start = time.time()
    from skopt.space import Real, Categorical, Integer
    from skopt.utils import use_named_args

    reg = LGBMRegressor()
    space = [Integer(10, 300, name='n_estimators'),
             Integer(5, 20, name='max_depth'),
             Integer(20, 100, name='num_leaves'),
             Integer(10, 100, name='min_data_in_leaf'),
             Real(0.01, 0.3, name='learning_rate'),
             Real(0.5, 1, name='feature_fraction'),
             Real(0, 1, name='lambda_l1'),
             Real(0, 1, name='lambda_l2')]

    @use_named_args(space)
    def objective(**params):
        # print('更新超参数的值')
        ValueList = list(params.values())
        params.update({'n_estimators': int(ValueList[0]), 'max_depth': int(ValueList[1]),
                       'num_leaves': int(ValueList[2]), 'min_data_in_leaf': int(ValueList[3])})

        reg.set_params(**params)

        roi_total_hpo = []
        popt_total_hpo = []
        rus_opt = RandomUnderSampler()
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.1, random_state=42)

        y_train_index = X_train.index
        y_opt_us = clean_data['bug'][y_train_index]
        x_train = X_train.values
        X_opt, y_opt = rus_opt.fit_resample(x_train, y_opt_us)
        X_opt_df = pd.DataFrame(X_opt, columns=X_df.columns)
        y_opt_df = pd.DataFrame(y_opt, columns=['bug'])
        X_opt_df['codechurn'] = (X_opt_df['la'] + X_opt_df['ld']) * X_opt_df['lt']
        X_opt_df['nf_nl'] = (X_opt_df['nf'] - X_opt_df['nf'].min(axis=0)) / (
                X_opt_df['nf'].max(axis=0) - X_opt_df['nf'].min(axis=0))
        coe_dif_os = X_opt_df['nf_nl'] ** X_opt_df['entropy']
        X_opt_df['effort'] = coe_dif_os * X_opt_df['codechurn']
        y_opt_dd = y_opt_df['bug'] / X_opt_df['effort']
        y_opt_dd = np.asarray(y_opt_dd).ravel()

        reg.fit(X_opt, y_opt_dd)
        y_new = reg.predict(X_test)
        y_act = clean_data['bug'][X_test.index]
        y_act = y_act.reset_index(drop=True)
        measure = clean_data['effort'][X_test.index]
        measure = measure.reset_index(drop=True)
        roi = computeAcc(y_act, y_new, measure)
        roi_total_hpo.append(roi[1])
        popt = computePopt(y_act, y_new, measure)
        popt_total_hpo.append(popt)

        acc = sum(roi_total_hpo) / len(roi_total_hpo)
        popt = sum(popt_total_hpo) / len(popt_total_hpo)

        return [-acc, -popt]

    pop = 300
    MaxIter = 500
    dim = 8
    lb = np.array([10, 5, 20, 10, 0.01, 0.5, 0, 0])
    ub = np.array([500, 20, 100, 100, 0.3, 1, 1, 1])
    fobj = objective
    GbestScore, GbestPositon, Curve = SMA_MO.SMA(pop, dim, lb, ub, MaxIter, fobj)
    tuningtime = time.time() - start
    print('The result of %s is as follow: ' % i)
    acc2 = -GbestScore[0]
    popt2 = -GbestScore[1]
    s = pd.Series([i, acc1, popt1, acc2, popt2, tuningtime], index=dfResult.columns)
    dfResult = dfResult.append(s, ignore_index=True)

    p1 += 1
    plt.figure(p1)
    plt.style.use('default')
    plt.plot(Curve[:, 0], 'b-', linewidth=2)
    plt.plot(Curve[:, 1], 'g-', linewidth=2)
    plt.xlabel('Iteration Times', fontsize='medium')
    plt.ylabel("Optimization Value", fontsize='medium')
    plt.grid()
    plt.title('WCMS under cross validation scenario', fontsize='large')
    plt.savefig('./result/FirstPaper_CrossValidation_LGBM_MO_' + i + getCurrentDateTime() + '.jpg', dpi=300, bbox_inches='tight')

    dfCurve = pd.DataFrame(Curve, columns=['ACC', 'Popt'])
    curveFileName = './result/FirstPaper_CrossValidation_LGBM_MO_' + i + getCurrentDateTime() + '.xls'
    dfCurve.to_excel(curveFileName, index=False)

# generate excel file
filename = './result/FirstPaper_CrossValidation_LGBM_MO_' + getCurrentDateTime() + '.xls'
dfResult.to_excel(filename, index=False)
