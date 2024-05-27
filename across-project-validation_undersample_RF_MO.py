import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import SMA_MO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")

# Get current datetime
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
train_projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
test_projects = ["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]
datapath = r'./data/'
dfResult = pd.DataFrame(columns=['project1', 'project2', 'acc1', 'popt1', 'acc2', 'popt2', 'tuningtime'])
p1 = 0
for i1 in train_projects:
    train_datafile = datapath + i1 + '.csv'
    train_data = pd.read_csv(train_datafile)
    train_data['codechurn'] = (train_data['la'] + train_data['ld']) * train_data['lt']
    train_clean_data = train_data[train_data['codechurn'] > 0]
    train_clean_data = train_clean_data.reset_index(drop=True)
    rfc = RandomForestRegressor(n_estimators=100)
    roi_total = []
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
    rfc.fit(X_resampled, y_resampled_dd)

    for i2 in test_projects:
        if i1 == i2:
            continue
        else:
            test_datafile = datapath + i2 + '.csv'
            test_data = pd.read_csv(test_datafile)
            test_data['codechurn'] = (test_data['la'] + test_data['ld']) * test_data['lt']
            test_clean_data = test_data[test_data['codechurn'] > 0]
            test_clean_data = test_clean_data.reset_index(drop=True)
            test_clean_data['nf_nl'] = (test_clean_data['nf'] - test_clean_data['nf'].min(axis=0)) / (
                    test_clean_data['nf'].max(axis=0) - test_clean_data['nf'].min(axis=0))
            test_coe_dif = test_clean_data['nf_nl'] ** test_clean_data['entropy']
            test_clean_data['effort'] = test_coe_dif * test_clean_data['codechurn']
            X_test_df = test_clean_data[
                ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 'rexp', 'sexp']]
            X_test = X_test_df.values
            y_new = rfc.predict(X_test)
            y_act = test_clean_data['bug']
            measure = test_clean_data['effort']
            roi = computeAcc(y_act, y_new, measure)
            popt = computePopt(y_act, y_new, measure)
            acc1 = roi[1]
            popt1 = popt
            start = time.time()

            from skopt.space import Real, Categorical, Integer
            from skopt.utils import use_named_args

            reg = RandomForestRegressor()
            space = [Integer(10, 500, name='n_estimators'),
                     Integer(5, 20, name='max_depth'),
                     Integer(1, 14, name='max_features'),
                     Integer(2, 11, name='min_samples_split'),
                     Integer(1, 11, name='min_samples_leaf'),
                     Categorical(['squared_error', 'absolute_error'], name='criterion')]

            # Define the objective function
            @use_named_args(space)
            def objective(**params):
                ValueList = list(params.values())
                params.update(
                    {'n_estimators': int(ValueList[0]), 'max_depth': int(ValueList[1]), 'max_features': int(ValueList[2]),
                     'min_samples_split': int(ValueList[3]), 'min_samples_leaf': int(ValueList[4])})
                if ValueList[5] <= 1:
                    params.update({'criterion': 'squared_error'})
                else:
                    params.update({'criterion': 'absolute_error'})

                reg.set_params(**params)
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
                rfc.fit(X_resampled, y_resampled_dd)
                y_new = rfc.predict(X_test)
                y_act = test_clean_data['bug']
                measure = test_clean_data['effort']
                roi = computeAcc(y_act, y_new, measure)
                acc = roi[1]
                popt = computePopt(y_act, y_new, measure)

                return [-acc, -popt]


            pop = 300
            MaxIter = 500
            dim = 6
            lb = np.array([10, 5, 1, 2, 1, 0])
            ub = np.array([500, 20, 14, 11, 11, 2])
            fobj = objective
            GbestScore, GbestPositon, Curve = SMA_MO.SMA(pop, dim, lb, ub, MaxIter, fobj)
            p1 += 0
            tuningtime = time.time() - start
            acc2 = -GbestScore[0]
            popt2 = -GbestScore[1]

            s = pd.Series([i1, i2, acc1, popt1, acc2, popt2, tuningtime], index=dfResult.columns)
            dfResult = dfResult.append(s, ignore_index=True)

            plt.figure(p1)
            plt.plot(Curve[:, 0], 'b-', linewidth=2)
            plt.plot(Curve[:, 1], 'g-', linewidth=2)
            plt.xlabel('Iteration Times', fontsize='medium')
            plt.ylabel("Optimization Value", fontsize='medium')
            plt.grid()
            plt.title('WCMS under across project scenario', fontsize='large')
            # plt.show()
            # 自动保存为文件
            plt.savefig('./result/FirstPaper_AcrossProject_Validation_RF_' + i1 + '_' + i2 + getCurrentDateTime() + '.jpg', dpi=300, bbox_inches='tight')

# generate excel file
filename = './result/FirstPaper_AcrossProject_Validation_RF_MO_' + getCurrentDateTime() + '.xls'
dfResult.to_excel(filename, index=False)
