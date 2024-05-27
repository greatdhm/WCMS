import numpy as np
import copy as copy

def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]

    return X

def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 2])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[[i, i], :] = X[index[i], :]
    return Xnew

def SMA(pop, dim, lb, ub, MaxIter, fun):
    z = 0.03
    X = initialization(pop, ub, lb, dim)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 2])
    W = np.zeros([pop, dim])
    mo = 2
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S = bestFitness - worstFitness + 10E-8
        for i in range(pop):
            if i < pop / 2:
                w_tmp1 = np.zeros((mo, dim))
                for j in range(mo):
                    w_tmp1[j] = 1 + np.random.random([1, dim]) * np.log10((bestFitness[j] - fitness[i][j]) / S[j] + 1)
                W[i, :] = np.amax(w_tmp1, axis=0)
            else:
                w_tmp2 = np.zeros((mo, dim))
                for j in range(mo):
                    w_tmp2[j] = 1 - np.random.random([1, dim]) * np.log10((bestFitness[j] - fitness[i][j]) / S[j] + 1)
                W[i, :] = np.amin(w_tmp2, axis=0)
        tt = -(t / MaxIter) + 1
        if tt != -1 and tt != 1:
            a = np.math.atanh(tt)
        else:
            a = 1
        b = 1 - t / MaxIter
        for i in range(pop):
            if np.random.random() < z:
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r < p.all():
                        X[i, j] = GbestPositon[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[0, j] * X[i, j]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        if fitness[0].any() <= GbestScore.any():
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
