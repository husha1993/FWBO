import warnings
import os
from copy import deepcopy

import numpy as np
import scipydirect
from DIRECT import solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import GPy

from design_optimize_alg.optimizer.BasicOptimizer import BasicOptimizer
from design_optimize_alg.surrogates.models.model_list import model_list
from design_optimize_alg.optimizer.helper.utils import LatinDesign
from design_optimize_alg.test_functions.mfFunction import mfFunction
from design_optimize_alg.acquisitions.CMFEntropySearch import CMFEntropySearch
from design_optimize_alg.optimizer.helper.write_model_configs import write_model_configs
from helpers.helpers import set_random_seeds, logfile


class CmfBayesOptimizer(BasicOptimizer):
    # Continuous multi-fidelity Bayesian Optimizer
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
        self.Y = []
        self.Fidelity = []
        self.costMarker = []
        self.itercost = []
        self.yMaxMarker = []

        self.Xtarget = []
        self.Ytarget = []
        self.ytargetMaxMarker = []
        self.Xtilta = []
        self.Ytilta = []
        self.ytiltaMaxMarker = []

        self.batch_idx = 0
        self.train_GP_num = 0
        self.current_fidelity = None
        self.budget_current = 0
        self.yMax = -np.inf
        self.info = dict()

    def configure(self):
        config = self.config
        self.name = config['name']
        self.xbounds = np.array(config["xbounds"])
        self.zbounds = np.array(config["zbounds"])
        self.bounds = np.array(config["bounds"])
        self.xdim = self.xbounds.shape[1]
        self.zdim = self.zbounds.shape[1]
        self.x_ninit = config['x_ninit']
        self.z_ninit = config['z_ninit']
        self.ini_seeds = config['ini_seeds']
        self.method = 'direct'
        self.gp_update_interval = config["gp_update_interval"]
        self.warping_update_interval = config['warping_update_interval']

        self._write_model_config()
        self.config["acq"]["num_representer_points"] = 100 #min(self.config['GPRS']['k_dim'] * 10, 100)
        self.config["acq"]["target_fidelity_index"] = [i for i in range(self.bounds.shape[1] - self.zdim, self.bounds.shape[1])]
        self.config["acq"]["highest_fidelity"] = self.zbounds[1, :]
        self.config["acq"]["sampler"]["bounds"] = self.xbounds
        self.config["acq"]["sampler"]["dim"] = self.xbounds.shape[1]
        self.acq_name = self.config["acq"]["name"]
        self.yscale = 1

        if config['set_mff_target']:
            self.mff_target = deepcopy(self.mff)
        if config['set_mff_inference']:
            self.mff_inference = deepcopy(self.mff)

    def _write_model_config(self):
        xbounds = self.xbounds
        zbounds = self.zbounds
        self.config["GPRS"].update(write_model_configs(xbounds, zbounds, self.config['GPRS']))
        self.logger.debug("model config: self.config[GPRS]={}".format(self.config["GPRS"]))

    def _configure_acq(self):
        config = self.config["acq"]
        self.acquisition = CMFEntropySearch(self.gps, config, self.logger)

    def initializeGPs(self):
        set_random_seeds(self.ini_seeds)
        x_ninit = self.x_ninit
        x = LatinDesign(self.xbounds, x_ninit)
        self.logger.debug('init x shape={}'.format(x.shape))
        self.logger.debug('init x={}'.format(x))

        if self.config['ini_manner'] == 'latin_cross':
            z_ninit = self.z_ninit
            z = LatinDesign(self.zbounds, z_ninit)

            z = self._roundfidelity(z)
            z = z[np.where(z.sum(axis=1)!=sum(self.zbounds[1, :]))[0], :]

            self.logger.debug('init z shape={}'.format(z.shape))
            self.logger.debug('init z={}'.format(z))
            z_target = np.concatenate([self.zbounds[1:, :] for i in range(x.shape[0])])
            x_ = np.random.permutation(LatinDesign(self.xbounds, min(50, self.z_ninit)))[:z.shape[0], :]
            x = np.concatenate([x, x_])
            z = np.concatenate([z_target, z])
            self.logger.debug('final ini z={}'.format(z))
            self.logger.debug('final ini x={}'.format(x))
        else:
            raise NotImplementedError

        xz = np.concatenate((x, z), axis=1)
        self.xz_ninit = xz.shape[0]
        self.logger.debug("xz_ini cost={}".format(sum(self.mff.getCost(xz[:, -self.zdim:]))))
        self.logger.debug("xz_init shape={}".format(xz.shape))
        self.logger.debug("xz_init={}".format(xz))
        y = self.evaluate(xz[:, :-self.zdim], xz[:, -self.zdim:], resources=self.mff.getCost(xz[:, -self.zdim:], mode="c"), mffname='mff')

        self.batch_idx += 1
        self.logger.info("y_init:{}".format(y))
        self.logger.info("budget used after initialization:{}".format(self.budget_current))
        self.logger.info("budget used after initialization(from mff):{}".format(self.mff.currentBudget))
        self.logger.debug("y_init shape={}".format(y.shape[0]))

        #init a surrogate
        self.gps = model_list[self.config['surrogate_name']](X_init=xz, Y_init=y, z_min=self.zbounds[0, :], z_max=self.zbounds[1, :], config=self.config['GPRS'])
        self.logger.debug('**************Check the init surrogate={}'.format(self.gps.model))
        self.gps.update()
        self.gps.fix_warping_functions()
        self.verbose()
        self.logger.debug("end model initialization")
        self.iterations += 1
        self.record(xz[:, :-self.zdim], xz[:, -self.zdim:], y)

    def verbose(self):
        self.logger.debug("**************StartSurrogateCheck*********************************")
        self.logger.debug("gps model={}".format(self.gps.model))
        self.logger.debug("gps model length scale={}".format(self.gps.model['.*lengthscale']))
        self.logger.debug("gps.model.X.shape ={}, _X.shape={}, Y.shape={}, _Y.shape={}".format(\
            self.gps.model.X.shape, self.gps._X.shape, self.gps.model.Y.shape, self.gps._Y.shape))
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
            if self.iterations % self.warping_update_interval == 0:
                self.gps.unfix_warping_functions()
            else:
                self.gps.fix_warping_functions()
            xz = self.optimize_acq_f(n_iter=self.config['optniter'], method=self.method)
            self.logger.debug("new next to evaluate x = {}, z={}".format(xz[:, :-self.zdim], xz[:, -self.zdim:]))
            y = self.evaluate(xz[:, :-self.zdim], xz[:, -self.zdim:], resources=self.mff.getCost(xz[:, -self.zdim:], mode="c"), mffname='mff')
            self.logger.debug("new evaluate y = {}".format(y))

            self.record(xz[:, :-self.zdim], xz[:, -self.zdim:], y)
            self.gps.add_data(xz, y)
            self.batch_idx += 1

    def evaluate(self, x, fidelity, resources=None, mode="c", mffname='mff'):
        if mffname == 'mff':
            y = self.mff.eval(x, fidelity, resources, mode) * self.yscale
        elif mffname == 'mff_target':
            if not np.all(fidelity == self.zbounds[1, :]):
                raise ValueError
            y = self.mff_target.eval(x, fidelity, resources, mode, update=False)
        elif mffname == 'mff_inference':
            self.logger.debug('*******start mff_inference eval')
            if not np.all(fidelity == self.zbounds[1, :]):
                raise ValueError
            y = self.mff_inference.eval(x, fidelity, resources, mode, update=False)
        y = (-1) * y
        return y.reshape((-1, 1))

    def optimize_acq_f(self, n_iter=20, method='random'):
        self.logger.debug("optimizer for acq: {}".format(method))
        self.logger.debug("n_iter={}".format(n_iter))
        self.logger.debug("optninit={}".format(self.config))
        self.acquisition.update() #especially for information-theory based acq
        def obj_LBFGS(x):
            #self.logger.debug('input to obj_LBFGS={}'.format(x))
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
                # Find the minimum of minus the acquisition function

                res = minimize(obj_LBFGS,
                               x_try.reshape(1, -1),
                               bounds=self.reformat_bounds(self.bounds),
                               method="L-BFGS-B")
                if not res.success:
                    self.logger.debug('minimize is not successful and going to try another random init(x_seeds) for minimize')
                    continue

                # Store it if better than previous minimum(maximum).
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
        x_max[:, -self.zdim:] = self._roundfidelity(x_max[:, -self.zdim:])
        self.logger.debug('end optimizing acq_f, with x_max={}, self.acq_f(x_max)={}'.format(x_max, self.acq_f(x_max)))
        return np.clip(x_max, self.bounds[0, :], self.bounds[1, :]).reshape((1, -1))

    def _roundfidelity(self, fidelities):
        fidelities = fidelities.copy()
        flag = (abs(fidelities - self.zbounds[1, :]) < np.array([0.1 for i in range(self.zdim)]))
        r = flag[:, 0]
        for i in range(1, self.zdim):
            r *= flag[:, i]
        r = r.reshape((-1, 1)).astype(float)
        fidelities[np.where(r == 1)[0], :] = self.zbounds[1, :]
        return fidelities

    def checkUpdateGPs(self):
        return True if self.iterations % self.gp_update_interval == 0 else False

    def acq_f(self, xz):
        if len(xz.shape) == 1:
            xz = np.array([xz])
        z = xz[:, -self.zdim:]
        gain_per_cost = self.acquisition.evaluate(xz)/self.mff.getCost(z, mode='c')
        #self.logger.debug('xz={}, gain_per_cost={}'.format(xz, gain_per_cost))
        return gain_per_cost

    def optimize_posterior_mean(self, n_iter=20, method='random'):
        self.logger.debug('optimizer for posterior mean at the highest fidelity'.format(method))
        self.logger.debug('n_iter={}'.format(n_iter))

        def obj_LBFGS(x):
            x = np.reshape(x, (-1, self.xdim))
            ztarget = np.tile(self.zbounds[1, :], (x.shape[0], 1))
            xz = np.concatenate([x, ztarget], axis=1)
            mean, _ = self.gps.predict(xz)
            mean = -mean
            return -mean

        def obj_DIRECT(x, u):
            x = np.reshape(x, (-1, self.xdim))
            ztarget = np.tile(self.zbounds[1, :], (x.shape[0], 1))
            xz = np.concatenate([x, ztarget], axis=1)
            mean, _ = self.gps.predict(xz)
            mean = -mean
            return -mean, 0

        x_tries = np.random.uniform(self.xbounds[0, :], self.xbounds[1, :], size=(self.config['optninit'], self.xbounds.shape[1]))

        if method == 'random':
            x_seeds = np.random.uniform(self.xbounds[0, :], self.xbounds[1, :], size=(n_iter, self.xbounds.shape[1]))
            ys = -obj_LBFGS(x_tries)
            x_max = x_tries[ys.argmax()].reshape((1, -1))
            max_acq = ys.max()
            self.logger.debug('max_acq from random={}'.format(max_acq))
            for x_try in x_seeds:
                res = minimize(obj_LBFGS, x_try.reshape(1, -1), bounds=self.reformat_bounds(self.xbounds), method='L-BFGS-B')
                if not res.success:
                    self.logger.debug('minimize is not successful and going to try another random init(x_seeds) for minimize')
                    continue
                if max_acq is None or -res.fun[0] > max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
                    self.logger.debug('use a result from minimize whose max_acq={}'.format(max_acq))
        elif method == 'direct':
            x, _, _ = solve(obj_DIRECT, self.xbounds[0, :], self.xbounds[1, :], maxf=1000, logfilename=logfile)
            x = minimize(obj_LBFGS, x, bounds=self.reformat_bounds(self.xbounds), method='L-BFGS-B').x
            x_max = x
        else:
            raise NotImplementedError
        x_max = x_max.reshape((1, -1))
        self.logger.debug('end optimizing posterior mean, with x_max={}, posterior_mean(x_max)={}'.format(x_max, -obj_LBFGS(x_max)))
        return np.clip(x_max, self.xbounds[0, :], self.xbounds[1, :]).reshape((1, -1))

    def record(self, x, z, y):
        y = y/self.yscale
        self.X.append(x)
        y = (-1) * y
        self.Y.append(y)
        self.Fidelity.append(z)
        self.costMarker.append(deepcopy(self.mff.currentBudget))
        self.yMaxMarker.append(deepcopy(self.mff.yMax))
        self.itercost.append(sum(self.mff.getCost(z, mode="c")))
        if not len(self.X) == len(self.Fidelity) == len(self.Y) == self.iterations:
            raise ValueError('recording error')
        self.writer.add_scalar('current_y_VS_iterations', self.Y[-1][-1], self.iterations)
        self.writer.add_scalar('current_cost_VS_iterations', self.itercost[-1], self.iterations)
        self.writer.add_scalar('current_yMax_VS_iterations', self.yMaxMarker[-1], self.iterations)
        self.writer.add_scalar('current_yMax_VS_acc_cost', self.yMaxMarker[-1], self.costMarker[-1])
        self.writer.add_scalar('current_y_VS_acc_cost', self.Y[-1][-1], self.costMarker[-1])
        for i in range(self.mff.xOpt.shape[0]):
            self.writer.add_scalar('current_xOpt_{}_vs_iterations'.format(i), self.mff.xOpt[i], self.iterations)

        if self.mff_target is not None and self.mff.currentBudget > self.mff_target.currentBudget:
            xtarget = x[-1:, :]
            ztarget = np.tile(self.zbounds[1, :], (xtarget.shape[0], 1))
            ytarget = self.evaluate(xtarget, ztarget, resources=self.mff_target.getCost(ztarget, mode="c"), mffname='mff_target')
            ytarget = (-1) * ytarget
            self.Xtarget.append(xtarget)
            self.Ytarget.append(ytarget)
            self.ytargetMaxMarker.append(deepcopy(self.mff_target.yMax))
            self.writer.add_scalar('current_ytarget_vs_iterations', self.Ytarget[-1][-1], self.iterations)
            self.writer.add_scalar('current_ytarget_vs_acc_cost', self.Ytarget[-1][-1], self.costMarker[-1])
            self.writer.add_scalar('current_ytargetMax_vs_iterations', self.ytargetMaxMarker[-1], self.iterations)
            self.writer.add_scalar('current_ytargetMax_vs_acc_cost', self.ytargetMaxMarker[-1], self.costMarker[-1])
            self.writer.add_scalar('acc_cost_vs_iterations', self.costMarker[-1], self.iterations)
            for i in range(self.mff_target.xOpt.shape[0]):
                self.writer.add_scalar('current_xOpt_{}_frommfftarget_vs_iterations'.format(i), self.mff_target.xOpt[i], self.iterations)

        xtilta = self.optimize_posterior_mean(n_iter=self.config['optniter'], method=self.method)
        self.Xtilta.append(xtilta)
        if self.mff_inference is not None and self.mff.currentBudget > self.mff_inference.currentBudget:
            ztarget = np.tile(self.zbounds[1, :], (xtilta.shape[0], 1))
            ytilta = self.evaluate(xtilta, ztarget, resources=self.mff_inference.getCost(ztarget, mode='c'), mffname='mff_inference')
            ytilta = (-1) * ytilta
            self.Ytilta.append(ytilta)
            self.ytiltaMaxMarker.append(deepcopy(self.mff_inference.yMax))
            self.writer.add_scalar('current_ytilta_vs_iterations', self.Ytilta[-1][-1], self.iterations)
            self.writer.add_scalar('current_ytilta_vs_acc_cost', self.Ytilta[-1][-1], self.costMarker[-1])
            self.writer.add_scalar('current_ytiltaMax_vs_iterations', self.ytiltaMaxMarker[-1], self.iterations)
            self.writer.add_scalar('current_ytiltaMax_vs_acc_cost', self.ytiltaMaxMarker[-1], self.costMarker[-1])
            self.writer.add_scalar('acc_cost_vs_iterations_infer', self.costMarker[-1], self.iterations)
            for i in range(self.mff_inference.xOpt.shape[0]):
                self.writer.add_scalar('current_xOpt_{}_frommffinference_vs_iterations'.format(i), self.mff_inference.xOpt[i], self.iterations)

        if self.iterations % self.config['plot_interval'] == 0:
            self._plot()
        self.logger.debug("***********************StartRecordInfo at Iteration={}***************".format(self.iterations))
        self.logger.debug("config['op_algorithm_config']['plt_dir']={}".format(self.config['plt_dir']))
        self.logger.debug("x={},z={},y={}".format(x, z, y))
        self.logger.debug("cost={}".format(sum(self.mff.getCost(z, mode="c"))))
        self.logger.debug('-------XYZ')
        self.logger.debug("total eval={}".format(np.concatenate(self.X).shape[0]))
        for i in range(np.concatenate(self.X).shape[0]-1, max(np.concatenate(self.X).shape[0]-30-1, -1), -1):
            self.logger.debug("X[{0}]={1},Z[{0}]={2},Y[{0}]={3}, Cost[{0}]={4}, Cost[{0}]/budget used={5}%".format(\
                i, np.concatenate(self.X)[i, :], np.concatenate(self.Fidelity)[i, :], np.concatenate(self.Y)[i, :], sum(self.mff.getCost(np.concatenate(self.Fidelity)[i, :], mode='c')), \
                              sum(self.mff.getCost(np.concatenate(self.Fidelity)[i, :], mode='c'))/self.mff.currentBudget*100))
        self.logger.debug('-------xtilta')
        for i in range(np.concatenate(self.Xtilta).shape[0]-1, max(np.concatenate(self.Xtilta).shape[0]-30-1, -1), -1):
            self.logger.debug("X[{0}]={1}".format(i, np.concatenate(self.Xtilta)[i, :]))
        self.logger.debug('-------')
        self.logger.debug("budget used={}".format(self.budget_current))
        self.logger.debug("budget used(from mff)={}".format(self.mff.currentBudget))
        self.logger.debug("budget used(from mff)/self.budget={}%".format(self.mff.currentBudget/self.budget *100))
        self.logger.debug("yMax={}".format(self.yMax))
        self.logger.debug("(from mff) yMax={},xOpt={},zOpt={} ".format(self.mff.yMax, self.mff.xOpt, self.mff.zOpt))
        self.logger.debug('(from mff) yMaxIte={}, yMaxBudget={}'.format(self.mff.yMaxIte, self.mff.yMaxBudget))
        if self.mff_target is not None:
            self.logger.debug('(from mfftarget) budget used={}'.format(self.mff_target.currentBudget))
            self.logger.debug("(from mfftarget) yMax={}, xOpt={}, zOpt={}".format(self.mff_target.yMax, self.mff_target.xOpt, self.mff_target.zOpt))
            self.logger.debug('(from mfftarget) yMaxIte={}, yMaxBudget={}, yMaxBudget(from self.cost)={}'.format(self.mff_target.yMaxIte, self.mff_target.yMaxBudget, self.costMarker[int(self.mff_target.yMaxIte)-1]))

        if self.mff_inference is not None:
            self.logger.debug('(from mffinference) budget used={}'.format(self.mff_inference.currentBudget))
            self.logger.debug("(from mffinference) yMax={}, xOpt={}, zOpt={}".format(self.mff_inference.yMax, self.mff_inference.xOpt, self.mff_inference.zOpt))
            self.logger.debug('(from mffinference) yMaxIte={}, yMaxBudget={}, yMaxBudget(from self.cost)={}'.format(self.mff_inference.yMaxIte, self.mff_inference.yMaxBudget, self.costMarker[int(self.mff_inference.yMaxIte) - 1]))
        self.logger.debug("***********************EndRecordInfo at Iteration={}*******************".format(self.iterations))

    def _plot(self):
        savefigdir = os.path.join(self.config["plt_dir"], self.config['name'] + '_iter=' + str(self.iterations))
        savenpdir = os.path.join(self.config["plt_dir"], 'figdata', self.config['name'] + '_iter=' + str(self.iterations))
        if not os.path.exists(savenpdir):
            os.makedirs(savenpdir)

        fig = plt.figure(figsize=(15, 15), constrained_layout=True)
        gs = fig.add_gridspec(5, 5)
        fig.add_subplot(gs[1, 2:])
        plt.plot(np.array(self.costMarker), np.array(self.ytargetMaxMarker), color='red')
        plt.scatter(np.array(self.costMarker), np.array(self.ytargetMaxMarker), c='red')
        plt.xlabel('accumulated cost')
        plt.ylabel('best y')
        plt.title('best ytarget VS accumulated cost, best y={}'.format(self.ytargetMaxMarker[-1]))

        fig.add_subplot(gs[0, :])
        plt.plot(np.array(range(1, 1+len(self.itercost))), np.array(self.itercost), color='red')
        plt.scatter(np.array((range(1, 1+len(self.itercost)))), np.array(self.itercost), c='red')
        plt.xlabel('iter')
        plt.ylabel('query-cost')
        plt.title('query cost VS iteration, current_iter_cost={}'.format(self.itercost[-1]))

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
        self.logger.debug('finish saving figure to = {}'.format(savefigdir))

        np.save(os.path.join(savenpdir, 'costMarker'), np.array(self.costMarker))
        np.save(os.path.join(savenpdir, 'yMaxMarker'), np.array(self.yMaxMarker))
        np.save(os.path.join(savenpdir, 'itercost'), np.array(self.itercost))
        np.save(os.path.join(savenpdir, 'Y'), np.array(self.Y))
        self.logger.debug('finish saving figure data to = {}'.format(savenpdir))
        plt.close()

    def reformat_bounds(self, bounds):
        assert len(bounds) == 2, "unexpected number of bounds"
        return list(zip(*bounds))





