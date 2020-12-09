import numpy as np
from copy import deepcopy


class mfFunction(object):
    def __init__(self, mfs, xbounds, zbounds=None, costs=None, costmodel=None, optx=None):
        self.mfs = mfs
        self.xbounds = xbounds
        self.zbounds = zbounds
        self.costmodel = costmodel
        self.costs = costs
        self.optx = optx

        self.currentBudget = 0
        self.currentIte = 0
        self.yMax = -np.inf
        self.yMaxIte = 0
        self.yMaxBudget = 0
        self.xOpt = None
        self.zOpt = None

    def eval(self, x, i, resources=None, mode="c", update=True, record=True):
        if mode == "discrete" or mode == 'd':
            y = self.mfs[i-1](x, i, resources)
            self.currentBudget += self.getCost(i, mode='d')*x.shape[0]
            if len(self.costs) == 1 and len(self.mfs) == 1 and y.max() > self.yMax:
                self.xOpt = x
                self.yMax = y.max()
            return self.mfs[i-1](x, i, resources)
        else:
            y = self.mfs(x, i, resources, updatetracker=update)
            if record:
                self.currentBudget += sum(self.getCost(i, mode='c'))
                self.currentIte += x.shape[0]
                _r, _flag = self._recordyMax(i)
                if _flag:
                    temp_y = np.multiply(y, _r)
                    temp_y[temp_y == np.inf] = -np.inf
                    if self.yMax < temp_y.max():
                        self.yMax = temp_y.max()
                        self.yMaxIte = self.currentIte
                        self.yMaxBudget = deepcopy(self.currentBudget)
                        idx, _ = np.unravel_index(temp_y.argmax(), temp_y.shape)
                        self.xOpt = x[idx, :]
                        self.zOpt = i[idx, :]
            return y

    def _recordyMax(self, fidelities):
        flag = (abs(fidelities - self.zbounds[1, :]) < np.array([0.0001 for i in range(self.zbounds.shape[1])]))
        r = flag[:, 0]
        for i in range(1, self.zbounds.shape[1]):
            r *= flag[:, i]
        r = r.reshape((-1, 1)).astype(float)
        flag = r.any() > 0
        r[r == 0] = -np.inf
        return r, flag

    def getCost(self, i, mode="c"):
        # compute costs needed for each fidelity
        if mode == "discrete" or mode == 'd':
            return self.costs[i-1]
        else:
            return self.costmodel(i)
