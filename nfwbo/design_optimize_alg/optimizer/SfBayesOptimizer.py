import os
from copy import deepcopy

import numpy as np
import GPy
from scipy.optimize import minimize
import scipydirect
from DIRECT import solve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter

from design_optimize_alg.optimizer.BasicOptimizer import BasicOptimizer
from design_optimize_alg.surrogates.models.model_list import model_list
from design_optimize_alg.optimizer.helper.utils import LatinDesign
from design_optimize_alg.test_functions.mfFunction import mfFunction
from design_optimize_alg.acquisitions.EntropySearch import EntropySearch
from helpers.helpers import set_random_seeds, logfile


class SfBayesOptimizer(BasicOptimizer):
    # Single fidelity Bayesian Optimizer
    def __init__(self, logger, writer, obj_f, bounds, budget, config):
        super().__init__()
        self.logger = logger
        self.writer = writer
        self.mff = obj_f
        self.mff_target = None
        self.mff_inference = None

        self.bounds = np.array(bounds)
        self.budget = budget

        self.config = config
        self.configure()

        self.iterations = 0
        self.gps = None
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
        self.train_GP_num = 0
        self.budget_current = 0
        self.yMax = -np.inf
        self.info = dict()

    def configure(self):
        config = self.config
        self.name = config['name']
        self.xbounds = np.array(config['xbounds'])
        self.zbounds = np.array(config['zbounds'])
        if np.any(self.zbounds[0, :] != self.zbounds[1, :]):
            raise ValueError("config wrong")
        self.bounds = np.array(config["bounds"])
        self.xdim = self.xbounds.shape[1]
        self.zdim = self.zbounds.shape[1]
        self.x_ninit = config['x_ninit']
        self.ini_seeds = config['ini_seeds']
        self.method = 'direct'
        self.gp_update_interval = config['gp_update_interval']

        self.acq_name = self.config["acq"]["name"]
        if self.acq_name == 'ES':
            self.config["acq"]["num_representer_points"] = 100 #min(self.xdim * 50, 100)
            self.config["acq"]["sampler"]["bounds"] = self.bounds
            self.config["acq"]["sampler"]["dim"] = self.bounds.shape[1]
        elif self.acq_name == 'MES':
            self.config["acq"]['bounds'] = self.bounds
            self.config['acq']['grid_size'] = 10000 * self.bounds.shape[1]

        if config['set_mff_target']:
            self.mff_target = deepcopy(self.mff)
        if config['set_mff_inference']:
            self.mff_inference = deepcopy(self.mff)

    def _configure_acq(self):
        if self.acq_name == 'ES':
            config = self.config["acq"]
            self.acquisition = EntropySearch(self.gps, config, self.logger)
        elif self.acq_name == 'MES':
            config = self.config['acq']
            self.acquisition = MaxValueEntropySearch(self.gps, config, self.logger)

    def initializeGPs(self):
        set_random_seeds(self.ini_seeds)
        x_ninit = self.x_ninit
        x = LatinDesign(self.xbounds, x_ninit)
        self.logger.debug('init x shape ={}'.format(x.shape))
        self.logger.debug('init x={}'.format(x))
        z = np.tile(self.zbounds[1, :], (x.shape[0], 1))
        y = self.evaluate(x, z, resources=self.mff.getCost(z, mode="c"), mffname='mff')

        self.batch_idx += 1
        self.logger.debug('y_init:{}'.format(y))
        self.logger.debug("budget used after initialization:{}".format(self.budget_current))
        self.logger.debug("budget used after initializaiton(from mff):{}".format(self.mff.currentBudget))
        self.logger.debug('y_init shape={}'.format(y.shape[0]))

        # init a surrogate
        self.gps = model_list[self.config['surrogate_name']](X_init=x, Y_init=y, config=self.config['GPRS'])
        self.logger.debug('********************check the init surrogate={}'.format(self.gps.model))
        self.gps.update()
        self.verbose()
        self.logger.debug("end model initialization")
        self.iterations += 1
        self.record(x, y, z)

    def verbose(self):
        self.logger.debug("**************StartSurrogateCheck*********************************")
        self.logger.debug("gps model={}".format(self.gps.model))
        self.logger.debug("gps model length scale={}".format(self.gps.model['.*lengthscale']))
        self.logger.debug("gps.model.X.shape ={}, _X.shape={}, Y.shape={}, _Y.shape={}".format(self.gps.model.X.shape, \
                                                 self.gps._X.shape, self.gps.model.Y.shape, self.gps._Y.shape))
        self.logger.debug("**************EndSurrogateCheck*********************************")

    def optimize(self):
        self.logger.debug("start optimization")
        if self.iterations == 0:
            self.initializeGPs()
            self._configure_acq()
        set_random_seeds(self.config['seeds'])
        while self.mff.currentBudget < self.budget:
            if self.checkUpdateGPs():
                self.gps.update()
                self.verbose()
            self.iterations += 1
            x = self.optimize_acq_f(n_iter=self.config['optniter'], method=self.method)
            self.logger.debug('new next to evaluate x={}'.format(x))
            z = np.tile(self.zbounds[1, :], (x.shape[0], 1))
            y = self.evaluate(x, z, resources=self.mff.getCost(z, mode="c"), mffname='mff')
            self.logger.debug("new evaluate y= {}".format(y))

            self.record(x, y, z)
            self.gps.add_data(x, y)
            self.batch_idx += 1

    def evaluate(self, x, fidelity, resources=None, mode='c', mffname='mff'):
        #set_random_seeds(1)
        if mffname == 'mff':
            y = self.mff.eval(x, fidelity, resources, mode)
        elif mffname == 'mff_target':
            if not np.all(fidelity == self.zbounds[1, :]):
                raise ValueError
            y = self.mff_target.eval(x, fidelity, resources, mode, update=False)
        elif mffname == 'mff_inference':
            self.logger.debug('*******start mff_inference eval')
            if not np.all(fidelity == self.zbounds[1, :]):
                raise ValueError
            y = self.mff_inference.eval(x, fidelity, resources, mode, update=False)
        else:
            raise NotImplementedError
        if self.acq_name == 'ES' or self.acq_name == 'MES':
            y = (-1) * y
        return y.reshape((-1, 1))

    def optimize_acq_f(self, n_iter=20, method='random'):
        self.logger.debug("optimizer for acq: {}".format(method))
        self.logger.debug("n_iter={}".format(n_iter))
        self.logger.debug("optninit={}".format(self.config))

        if self.acq_name == 'ES' or self.acq_name == 'MES':
            self.acquisition.update() #especially for information-theory based acq

        def obj_LBFGS(x):
            return -self.acq_f(x)

        def obj_DIRECT(x, u):
            return -self.acq_f(x), 0

        x_tries = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(self.config['optninit'], self.bounds.shape[1]))
        if method == 'random':
            x_seeds = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(n_iter, self.bounds.shape[1]))
            ys = -obj_LBFGS(x_tries)
            x_max = x_tries[ys.argmax()].reshape((1, -1))
            max_acq = ys.max()
            self.logger.debug('max_acq from random={}'.format(max_acq))
            for x_try in x_seeds:
                res = minimize(obj_LBFGS, x_try.reshape(1, -1), bounds=self.reformat_bounds(self.bounds),
                               method='L-BFGS-B')
                if not res.success:
                    self.logger.debug('minimize is not successful and going to try another random init(x_seeds) for minimize')
                    continue

                if max_acq is None or -res.fun[0] > max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
                    self.logger.debug('use a result from minimize whose max_acq={}'.format(max_acq))

        elif method == 'direct':
            x, _, _ = solve(obj_DIRECT, self.bounds[0, :], self.bounds[1, :], maxf=1000, logfilename=logfile)
            x = minimize(obj_LBFGS, x, bounds=self.reformat_bounds(self.bounds), method='L-BFGS-B').x
            x_max = x
        else:
            raise NotImplementedError
        x_max = x_max.reshape((1, -1))
        self.logger.debug('end optimizing acq_f, with x_max={}, self.acq_f(x_max)={}'.format(x_max, self.acq_f(x_max)))
        return np.clip(x_max, self.bounds[0, :], self.bounds[1, :]).reshape((1, -1))

    def checkUpdateGPs(self):
        return True if self.iterations % self.gp_update_interval == 0 else False

    def acq_f(self, x, alpha=-1, v=.1, delta=.1):
        if self.acq_name == 'GP_UCB':
            x = np.reshape(x, (-1, self.xdim))
            mean, var = self.gps.predict(x)
            std = np.sqrt(var)
            if alpha is -1:
                alpha = np.sqrt(v * (2 * np.log((self.iterations** ((self.xdim / 2) + 2))
                                            * (np.pi ** 2) / (3 * delta))))
            return mean + (alpha * std)

        elif self.acq_name == 'ES' or self.acq_name == 'MES':
            if len(x.shape) == 1:
                x = np.array([x])
            entropy_reduction = self.acquisition.evaluate(x)
            #self.logger.debug('x={}, entropy_reduction={}'.format(x, entropy_reduction))
            return entropy_reduction

        else:
            raise NotImplementedError

    def optimize_posterior_mean(self, n_iter=20, method='random'):
        self.logger.debug("optimizer for posterior mean: {}".format(method))
        self.logger.debug("n_iter={}".format(n_iter))


        def obj_LBFGS(x):
            x = np.reshape(x, (-1, self.xdim))
            mean, _ = self.gps.predict(x)
            if self.acq_name == 'ES' or self.acq_name == 'MES':
                mean = -mean
            return -mean

        def obj_DIRECT(x, u):
            x = np.reshape(x, (-1, self.xdim))
            mean, _ = self.gps.predict(x)
            if self.acq_name == 'ES' or self.acq_name == 'MES':
                mean = -mean
            return -mean, 0

        x_tries = np.random.uniform(self.bounds[0, :], self.bounds[1, :],
                                    size=(self.config['optninit'], self.bounds.shape[1]))
        if method == 'random':
            x_seeds = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(n_iter, self.bounds.shape[1]))
            ys = -obj_LBFGS(x_tries)
            x_max = x_tries[ys.argmax()].reshape((1, -1))
            max_acq = ys.max()
            self.logger.debug('max_acq from random={}'.format(max_acq))
            for x_try in x_seeds:
                res = minimize(obj_LBFGS, x_try.reshape(1, -1), bounds=self.reformat_bounds(self.bounds),
                               method='L-BFGS-B')
                if not res.success:
                    self.logger.debug(
                        'minimize is not successful and going to try another random init(x_seeds) for minimize')
                    continue

                if max_acq is None or -res.fun[0] > max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
                    self.logger.debug('use a result from minimize whose max_acq={}'.format(max_acq))

        elif method == 'direct':
            x, _, _ = solve(obj_DIRECT, self.bounds[0, :], self.bounds[1, :], maxf=1000, logfilename=logfile)
            x = minimize(obj_LBFGS, x, bounds=self.reformat_bounds(self.bounds), method='L-BFGS-B').x
            x_max = x
        else:
            raise NotImplementedError
        x_max = x_max.reshape((1, -1))
        self.logger.debug('end optimizing posterior mean, with x_max={}, posteror_mean(x_max)={}'.format(x_max, -obj_LBFGS(x_max)))
        return np.clip(x_max, self.bounds[0, :], self.bounds[1, :]).reshape((1, -1))

    def record(self, x, y, z):
        self.X.append(x)
        if self.acq_name == 'ES' or self.acq_name == 'MES':
            y = (-1) * y
        self.Y.append(y)
        self.costMarker.append(deepcopy(self.mff.currentBudget))
        self.yMaxMarker.append(deepcopy(self.mff.yMax))
        self.itercost.append(sum(self.mff.getCost(z, mode="c")))
        if not len(self.X) == len(self.Y) == self.iterations:
            raise ValueError('recording error')
        self.writer.add_scalar('current_y_VS_iterations', self.Y[-1][-1], self.iterations)
        self.writer.add_scalar('current_cost_VS_iterations', self.itercost[-1], self.iterations)
        self.writer.add_scalar('current_yMax_VS_acc_cost', self.yMaxMarker[-1], self.costMarker[-1])
        for i in range(self.mff.xOpt.shape[0]):
            self.writer.add_scalar('current_xOpt_{}_vs_iterations'.format(i), self.mff.xOpt[i], self.iterations)

        if self.mff_target is not None:
            ytarget = y
            self.Ytarget.append(ytarget)
            self.ytargetMaxMarker.append(deepcopy(self.mff.yMax))
            self.writer.add_scalar('current_ytarget_vs_iterations', self.Ytarget[-1][-1], self.iterations)
            self.writer.add_scalar('current_ytarget_vs_acc_cost', self.Ytarget[-1][-1], self.costMarker[-1])
            self.writer.add_scalar('current_ytargetMax_vs_iterations', self.ytargetMaxMarker[-1], self.iterations)
            self.writer.add_scalar('current_ytargetMax_vs_acc_cost', self.ytargetMaxMarker[-1], self.costMarker[-1])
            self.writer.add_scalar('acc_cost_vs_iterations', self.costMarker[-1], self.iterations)
            for i in range(self.mff.xOpt.shape[0]):
                self.writer.add_scalar('current_xOpt_{}_frommfftarget_vs_iterations'.format(i), self.mff.xOpt[i], self.iterations)
        if self.mff_inference is not None:
            xtilta = self.optimize_posterior_mean(n_iter=self.config['optniter'], method=self.method)
            ztarget = np.tile(self.zbounds[1, :], (xtilta.shape[0], 1))
            ytilta = self.evaluate(xtilta, ztarget, resources=self.mff_inference.getCost(ztarget, mode='c'), mffname='mff_inference')
            if self.acq_name == 'ES' or self.acq_name == 'MES':
                ytilta = (-1) * ytilta
            self.Xtilta.append(xtilta)
            self.Ytilta.append(ytilta)
            self.ytiltaMaxMarker.append(deepcopy(self.mff_inference.yMax))
            self.writer.add_scalar('current_ytilta_vs_iterations', self.Ytilta[-1][-1], self.iterations)
            self.writer.add_scalar('current_ytilta_vs_acc_cost', self.Ytilta[-1][-1], self.costMarker[-1])
            self.writer.add_scalar('current_ytiltaMax_vs_iterations', self.ytiltaMaxMarker[-1], self.iterations)
            self.writer.add_scalar('current_ytiltaMax_vs_acc_cost', self.ytiltaMaxMarker[-1], self.costMarker[-1])
            self.writer.add_scalar('acc_cost_vs_iterations_infer', self.costMarker[-1], self.iterations)
            for i in range(self.mff_inference.xOpt.shape[0]):
                self.writer.add_scalar('current_xOpt_{}_frommffinference_vs_iterations'.format(i), self.mff_inference.xOpt[i],
                                       self.iterations)

        if self.iterations % self.config['plot_interval'] == 0:
            self._plot()
        self.logger.debug("***********************StartRecordInfo at Iteration={}***************".format(self.iterations))
        self.logger.debug("config['op_algorithm_config']['plt_dir']={}".format(self.config['plt_dir']))
        self.logger.debug('x={}, y={}'.format(x, y))
        self.logger.debug('cost={}'.format(sum(self.mff.getCost(z, mode="c"))))
        self.logger.debug('------------XY')
        for i in range(np.concatenate(self.X).shape[0]-1, max(np.concatenate(self.X).shape[0]-30-1, -1), -1):
            self.logger.debug("X[{0}]={1}, Y[{0}]={2}".format(i, np.concatenate(self.X)[i, :], np.concatenate(self.Y)[i, :]))
        if self.mff_inference is not None:
            self.logger.debug('-------xtilta')
            for i in range(np.concatenate(self.Xtilta).shape[0]-1, max(np.concatenate(self.Xtilta).shape[0]-30-1, -1), -1):
                self.logger.debug("X[{0}]={1}".format(i, np.concatenate(self.Xtilta)[i, :]))
        self.logger.debug('------------')
        self.logger.debug('budget used={}'.format(self.budget_current))
        self.logger.debug('budget used(from mff)={}'.format(self.mff.currentBudget))
        self.logger.debug("budget used(from mff)/self.budget={}%".format(self.mff.currentBudget/self.budget *100))
        self.logger.debug("yMax={}".format(self.yMax))
        self.logger.debug("(from mff) yMax={},xOpt={},zOpt={} ".format(self.mff.yMax, self.mff.xOpt, self.mff.zOpt))
        if self.mff_target is not None:
            self.logger.debug('(from mfftarget) yMax={}, xOpt={}, zOpt={}'.format(self.mff.yMax, self.mff.xOpt, self.mff.zOpt))
            self.logger.debug('(from mfftarget) yMaxIte={}, yMaxBudget={}'.format(self.mff.yMaxIte, self.mff.yMaxBudget))
        if self.mff_inference is not None:
            self.logger.debug('(from mffinference) yMax={}, xOpt={}, zOpt={}'.format(self.mff_inference.yMax, self.mff_inference.xOpt, self.mff_inference.zOpt))
            self.logger.debug('(from mffinference) yMaxIte={}, yMaxBudget={}'.format(self.mff_inference.yMaxIte, self.mff_inference.yMaxBudget))
        self.logger.debug("***********************EndRecordInfo at Iteration={}*******************".format(self.iterations))

    def _plot(self):
        savefigdir = os.path.join(self.config["plt_dir"], self.config['name'] + '_iter=' + str(self.iterations))
        savenpdir = os.path.join(self.config["plt_dir"], 'figdata', self.config['name'] + '_iter=' + str(self.iterations))
        if not os.path.exists(savenpdir):
            os.makedirs(savenpdir)

        fig = plt.figure(figsize=(15, 15), constrained_layout=True)
        gs = fig.add_gridspec(5, 5)
        f_ax1 = fig.add_subplot(gs[1, 2:])
        plt.plot(np.array(self.costMarker), np.array(self.yMaxMarker), color='red', label=self.config['name'])
        plt.scatter(np.array(self.costMarker), np.array(self.yMaxMarker), c='red')
        plt.xlabel('accumulated cost')
        plt.ylabel("best y")
        plt.title('best y VS accumulated cost, best y={}'.format(self.yMaxMarker[-1].item()))

        #plot ytilta-vs-acc_cost
        f_ax1 = fig.add_subplot(gs[3, 2:])
        plt.plot(np.array(self.costMarker), np.array(self.ytiltaMaxMarker))
        plt.scatter(np.array(self.costMarker), np.array(self.ytiltaMaxMarker))
        plt.xlabel('accumulated cost')
        plt.ylabel("current ytiltaMax")
        plt.title('current ytiltaMax VS acc_cost, and current ytiltaMax={}'.format(self.ytiltaMaxMarker[-1]))

        #plot ytilta-vs-iteration
        f_ax1 = fig.add_subplot(gs[4, :])
        plt.plot(np.array(range(len(self.Ytilta))), np.array(self.Ytilta).reshape(-1))
        plt.scatter(np.array(range(len(self.Ytilta))), np.array(self.Ytilta).reshape(-1))
        plt.xlabel('iteration')
        plt.ylabel("current ytilta")
        plt.title('current ytilta VS iteration, and current ytilta={}'.format(self.Ytilta[-1]))

        plt.savefig(savefigdir, bbox_inches='tight')
        self.logger.debug('finish saving figure to ={}'.format(savefigdir))

        np.save(os.path.join(savenpdir, 'costMarker'), np.array(self.costMarker))
        np.save(os.path.join(savenpdir, 'yMaxMarker'), np.array(self.yMaxMarker))
        np.save(os.path.join(savenpdir, 'itercost'), np.array(self.itercost))
        np.save(os.path.join(savenpdir, 'Y'), np.array(self.Y))
        self.logger.debug('finish saving figure data to = {}'.format(savenpdir))
        plt.close()

    def reformat_bounds(self, bounds):
        assert len(bounds) == 2, "unexpected number of bounds"
        return list(zip(*bounds))


