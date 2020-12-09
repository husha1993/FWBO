import os
from copy import deepcopy
import time

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from design_optimize_alg.optimizer.BasicOptimizer import BasicOptimizer
from design_optimize_alg.optimizer.helper.utils import LatinDesign
from design_optimize_alg.test_functions.mfFunction import mfFunction
from helpers.helpers import set_random_seeds


class RandomOptimizer(BasicOptimizer):
    def __init__(self, logger, writer, obj_f, bounds, budget, config):
        super().__init__()
        self.logger = logger
        self.writer = writer
        self.mff = obj_f
        self.bounds = np.array(bounds)
        self.budget = budget

        self.config = config
        self.configure()

        self.iterations = 0
        self.X = []
        self.costMarker = []
        self.itercost = []
        self.yMaxMarker = []

        self.Ytarget = []
        self.ytargetMaxMarker = []
        self.Xtilta = []
        self.Ytilta = []
        self.ytiltaMaxMarker = []

        self.batch_idx = 0
        self.budget_current = 0
        self.yMax = -np.inf
        self.info = dict()

    def configure(self):
        config = self.config
        self.name = config["name"]
        self.xbounds = np.array(config['xbounds'])
        self.zbounds = np.array(config['zbounds'])
        self.bounds = np.array(config['bounds'])
        self.xdim = self.xbounds.shape[1]
        self.zdim = self.zbounds.shape[1]
        self.x_ninit = config['x_ninit']
        self.ini_seeds = config['ini_seeds']
        self.seeds = config['seeds']

    def initialize(self):
        set_random_seeds(self.ini_seeds)
        x_ninit = self.x_ninit
        x = LatinDesign(self.xbounds, x_ninit)
        self.logger.debug('init x shape ={}'.format(x.shape))
        self.logger.debug('init x={}'.format(x))
        z = np.tile(self.zbounds[1, :], (x.shape[0], 1))
        y = self.evaluate(x, z, resources=self.mff.getCost(z, mode='c'))

        self.batch_idx += 1
        self.logger.debug('y_init:{}'.format(y))
        self.logger.debug('budget used after initialization:{}'.format(self.budget_current))
        self.logger.debug('budget used after initialization(from mff):{}'.format(self.mff.currentBudget))
        self.logger.debug('y_init shape={}'.format(y.shape[0]))
        self.logger.debug('end initialization')
        self.iterations += 1
        self.record(x, y, z)

    def optimize(self):
        self.logger.debug('start optimization')
        if self.iterations == 0:
            self.initialize()
        set_random_seeds(self.config['seeds'])
        while self.mff.currentBudget < self.budget:
            self.iterations += 1
            np.random.seed(int(time.time() *10 ** 8 - int(time.time()) *10 ** 8))
            x = np.random.uniform(low=self.xbounds[0, :], high=self.xbounds[1, :], size=(1, self.xdim))
            np.random.seed(self.seeds)
            self.logger.debug('new next to evaluate x={}'.format(x))
            z = np.tile(self.zbounds[1, :], (x.shape[0], 1))
            y = self.evaluate(x, z, resources=self.mff.getCost(z, mode='c'))
            self.logger.debug("new evaluate y={}".format(y))

            self.record(x, y, z)
            self.batch_idx += 1

    def evaluate(self, x, fidelity, resources=None, mode="c"):
        y = self.mff.eval(x, fidelity, resources, mode)
        return y.reshape((-1, 1))

    def record(self, x, y, z):
        self.X.append(x)
        self.Y.append(y)
        self.costMarker.append(deepcopy(self.mff.currentBudget))
        self.yMaxMarker.append(deepcopy(self.mff.yMax))
        self.itercost.append(sum(self.mff.getCost(z, mode='c')))
        if not len(self.X) == len(self.Y) == self.iterations:
            raise ValueError('recording error')
        self.writer.add_scalar('current_y_VS_iterations', self.Y[-1][-1], self.iterations)
        self.writer.add_scalar('current_cost_VS_iterations', self.itercost[-1], self.iterations)
        self.writer.add_scalar('current_yMax_VS_acc_cost', self.yMaxMarker[-1], self.costMarker[-1])

        ytarget = y
        self.Ytarget.append(ytarget)
        self.ytargetMaxMarker.append(deepcopy(self.mff.yMax))
        self.writer.add_scalar('current_ytarget_vs_iterations', self.Ytarget[-1][-1], self.iterations)
        self.writer.add_scalar('current_ytarget_vs_acc_cost', self.Ytarget[-1][-1], self.costMarker[-1])
        self.writer.add_scalar('current_ytargetMax_vs_iterations', self.ytargetMaxMarker[-1], self.iterations)
        self.writer.add_scalar('current_ytargetMax_vs_acc_cost', self.ytargetMaxMarker[-1], self.costMarker[-1])

        Xtilta = x
        ytilta = y
        self.Xtilta.append(Xtilta)
        self.Ytilta.append(ytilta)
        self.ytiltaMaxMarker.append(deepcopy(self.mff.yMax))
        self.writer.add_scalar('current_ytilta_vs_iterations', self.Ytilta[-1][-1], self.iterations)
        self.writer.add_scalar('current_ytilta_vs_acc_cost', self.Ytilta[-1][-1], self.costMarker[-1])
        self.writer.add_scalar('current_ytiltaMax_vs_iterations', self.ytiltaMaxMarker[-1], self.iterations)
        self.writer.add_scalar('current_ytiltaMax_vs_acc_cost', self.ytiltaMaxMarker[-1], self.costMarker[-1])

        if self.iterations % self.config['plot_interval'] == 0:
            self._plot()
        self.logger.debug("***********************StartRecordInfo at Iteration={}***************".format(self.iterations))
        self.logger.debug("config['op_algorithm_config']['plt_dir']={}".format(self.config['plt_dir']))
        self.logger.debug('x={}, y={}'.format(x, y))
        self.logger.debug('cost={}'.format(sum(self.mff.getCost(z, mode="c"))))
        self.logger.debug('------------XY')
        for i in range(np.concatenate(self.X).shape[0]-1, max(np.concatenate(self.X).shape[0]-30-1, 0), -1):
            self.logger.debug("X[{0}]={1}, Y[{0}]={2}".format(i, np.concatenate(self.X)[i, :], np.concatenate(self.Y)[i, :]))
        self.logger.debug('------------')
        self.logger.debug('budget used={}'.format(self.budget_current))
        self.logger.debug('budget used(from mff)={}'.format(self.mff.currentBudget))
        self.logger.debug("budget used(from mff)/self.budget={}%".format(self.mff.currentBudget/self.budget *100))
        self.logger.debug("yMax={}".format(self.yMax))
        self.logger.debug("(from mff) yMax={},xOpt={},zOpt={} ".format(self.mff.yMax, self.mff.xOpt, self.mff.zOpt))
        self.logger.debug('(from mff) yMaxIte={}, yMaxBudget={}'.format(self.mff.yMaxIte, self.mff.yMaxBudget))
        self.logger.debug("***********************EndRecordInfo at Iteration={}*******************".format(self.iterations))

    def _plot(self):
        plt.figure(figsize=(15, 15))
        plt.subplot(211)
        plt.plot(np.array(self.costMarker), np.array(self.yMaxMarker), label=self.config['name'])
        plt.scatter(np.array(self.costMarker), np.array(self.yMaxMarker))
        plt.xlabel('accumulated cost')
        plt.ylabel("best y")
        plt.title('best y VS accumulated cost, best y={}'.format(self.yMaxMarker[-1].item()))

        if len(self.Y) >= 2:
            plt.subplot(212)
            plt.plot(np.array(range(1, len(self.Y))), np.concatenate(self.Y)[self.x_ninit:], label=self.config['name'])
            plt.scatter(np.array(range(1, len(self.Y))), np.array(self.Y[1:]))
            plt.xlabel('iteration')
            plt.ylabel("current y")
            plt.title('current y VS iteration, and current y={}'.format(self.Y[-1]))

        savefigdir = os.path.join(self.config['plt_dir'], self.config['name'] + '_iter=' + str(self.iterations))
        plt.savefig(savefigdir, bbox_inches='tight')
        self.logger.debug('finish saving figure to ={}'.format(savefigdir))

        savenpdir = os.path.join(self.config["plt_dir"], 'figdata', self.config['name'] + '_iter=' + str(self.iterations))
        if not os.path.exists(savenpdir):
            os.makedirs(savenpdir)
        np.save(os.path.join(savenpdir, 'costMarker'), np.array(self.costMarker))
        np.save(os.path.join(savenpdir, 'yMaxMarker'), np.array(self.yMaxMarker))
        np.save(os.path.join(savenpdir, 'itercost'), np.array(self.itercost))
        np.save(os.path.join(savenpdir, 'Y'), np.array(self.Y))
        self.logger.debug('finish saving figure data to = {}'.format(savenpdir))
        plt.close()




