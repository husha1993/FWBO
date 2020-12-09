import numpy as np
import pyDOE


def LatinDesign(bounds, numInitpoints):
    X_design_aux = pyDOE.lhs(bounds.shape[1], numInitpoints, criterion='center')
    ones = np.ones((X_design_aux.shape[0], 1))

    lower_bound = np.asarray(bounds)[0, :].reshape(1, bounds.shape[1])
    upper_bound = np.asarray(bounds)[1, :].reshape(1, bounds.shape[1])
    diff = upper_bound - lower_bound

    initPoints = np.dot(ones, lower_bound) + X_design_aux * np.dot(ones, diff)

    return initPoints


