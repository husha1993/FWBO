import os

import numpy as np

from envs.pybulletevo.evoenvs import HalfCheetahUrdfEnv, AntEnv
from design_optimize_alg.optimizer.CmfBayesOptimizer import CmfBayesOptimizer
from design_optimize_alg.optimizer.SfBayesOptimizer import SfBayesOptimizer
from design_optimize_alg.optimizer.RandomOptimizer import RandomOptimizer
from design_optimize_alg.optimizer.CmaEsOptimizer import CmaEsOptimizer
from design_optimize_alg.optimizer.HpcBatchBayesOptimizer import HpcBatchBayesOptimizer
from RL.algorithm.PPO_vec_batch import PPOVecBatch
from RL.env.clEnv import CLEnv
from RL.env.subproc_vec_env import SubprocVecEnv
import warnings
warnings.filterwarnings('ignore')


def select_environment(env_name):
    if env_name == "HalfCheetahUrdfEnv":
        return HalfCheetahUrdfEnv
    elif env_name == 'AntEnv':
        return AntEnv
    else:
        raise ValueError("Environment class not found.")


def make_env(env_name, config, seed):
    return lambda : CLEnv(select_environment(env_name)(config, seed))


def make_vec_env(env_name, config, num_env):
    seeds = list(range(num_env))
    return SubprocVecEnv([make_env(env_name, config, seeds[i]) for i in range(num_env)])


def select_rl_alg(rl_name):
    if rl_name == "PPOVecBatch":
        return PPOVecBatch
    else:
        raise ValueError('RL method not fund.')


def config_optimizer(logger, writer, config, obj_f):
    opt_name = config['optimizer_name']
    if opt_name.startswith('CmfBayesOptimizer'):
        obj_f = obj_f()
        return CmfBayesOptimizer(logger, writer, obj_f, np.array(config['op_algorithm_config']['bounds']), config['op_algorithm_config']['budget'], config['op_algorithm_config'] )
    elif opt_name == "SfBayesOptimizer":
        obj_f = obj_f()
        return SfBayesOptimizer(logger, writer, obj_f, np.array(config['op_algorithm_config']['bounds']),config['op_algorithm_config']['budget'], config['op_algorithm_config'] )
    elif opt_name == 'randomOptimizer':
        obj_f = obj_f()
        return RandomOptimizer(logger, writer, obj_f, np.array(config['op_algorithm_config']['bounds']),config['op_algorithm_config']['budget'], config['op_algorithm_config'] )
    elif opt_name == 'cmaesOptimizer':
        obj_f = obj_f()
        return CmaEsOptimizer(logger, writer, obj_f, np.array(config['op_algorithm_config']['bounds']),config['op_algorithm_config']['budget'], config['op_algorithm_config'] )
    elif opt_name == 'hpcbboOptimizer':
        obj_f = obj_f()
        return HpcBatchBayesOptimizer(logger, writer, obj_f, np.array(config['op_algorithm_config']),config['op_algorithm_config']['budget'], config['op_algorithm_config'] )
    else:
        raise ValueError('Optimizer method not fund.')


def fidelity2epochs(z, e):
    base_epochs = 50
    if len(z.shape) == 0 or z.shape[0] == 1:
        return np.array(z * e + base_epochs).astype(int)
    else:
        return (z * e + base_epochs).astype(int)


def fidelity2timelimits(z, t):
    return int(z * t + 50)


class DesignOptimize(object):
    def __init__(self, logger, writer, config):
        self.logger = logger
        self.writer = writer
        self._config = config

        self.max_path_length = self._config['rl_algorithm_config']['algo_params']['max_path_length']
        self.num_eval_steps_per_epoch = self._config['rl_algorithm_config']['algo_params']['num_steps_per_eval']

        self.num_env = self._config["rl_algorithm_config"]['num_env']
        self.env = make_vec_env(self._config['env']['env_name'], self._config, self.num_env)

        self.optimizer = config_optimizer(self.logger, self.writer, self._config, self.design_obj_f)
        self._last_optimizer_batch_idx = -1

        self.current_batch_designs = []
        self._episode_counter = 0
        self._design_counter = 0
        self._design_tracker = []
        self.METRICS = {}

    def optimize(self):
        self.optimizer.optimize()

    def design_obj_f(self, designs=None, num_epochs=None, design_idxs=None, last_halving=True):
        if self._config["optimizer_name"].startswith("CmfBayesOptimizer") or self._config["optimizer_name"] == 'SfBayesOptimizer' or \
                self._config["optimizer_name"] == 'randomOptimizer' or self._config["optimizer_name"] == 'cmaesOptimizer' or self._config['optimizer_name'] == 'hpcbboOptimizer':
            from design_optimize_alg.test_functions.mfFunction import mfFunction

            def costmodel(fidelity):
                z = fidelity.reshape((-1, 2))
                epoch = fidelity2epochs(z[:, 0], self._config['op_algorithm_config']['max_num_epochs_per_design'])
                maxepoch = fidelity2epochs(np.array(self._config["op_algorithm_config"]["zbounds"])[1:, 0], self._config['op_algorithm_config']['max_num_epochs_per_design'])
                return epoch/maxepoch

            def f(designs, fidelities, resoures = None, updatetracker=True):
                '''
                fidelities:[N,T]
                '''
                if designs.shape[0] != fidelities.shape[0]:
                    raise ValueError('the num of designs and the num of fidelities should be the same')
                if self._config["force_manual_design"]:
                    designs = np.array([self._config['manual_design']] * designs.shape[0])
                _current_optimizer_batch_idx = self.optimizer.batch_idx

                self._define_rl_stuff(designs, fidelities)
                self.current_batch_designs = designs
                self._design_tracker.append([{'design_episodes': 0, 'return': []} for _ in range(designs.shape[0])])
                time_limits = self._config['op_algorithm_config']['time_limits']
                samples = self._config['op_algorithm_config']['max_num_epochs_per_design']
                design_idxs = range(designs.shape[0])
                Returns = np.zeros(designs.shape[0])

                for _, design_idx in enumerate(design_idxs):
                    # set design
                    design = designs[design_idx]
                    self.env.set_new_design(design)
                    self.env.reset()
                    # set T
                    time_limit = fidelity2timelimits(fidelities[design_idx, 1], time_limits)
                    self._rl_alg.changeTimeLimit(time_limit)
                    # set N
                    sample = fidelity2epochs(fidelities[design_idx, 0], samples)
                    self.policy_learning(design_idx, sample)

                    Return = self.inner_evaluate(design_idx)
                    Returns[design_idx] = Return
                    if self.optimizer.yMax < Return and \
                            abs(fidelities[design_idx, 0]-self._config["op_algorithm_config"]["zbounds"][1][0])<0.1 \
                            and abs(fidelities[design_idx, 1]-self._config["op_algorithm_config"]["zbounds"][1][1])<0.1:
                        self.optimizer.yMax = Return
                    self.optimizer.budget_current += self.optimizer.mff.getCost(fidelities[design_idx, :], mode="c")
                    if updatetracker:
                        self.update_design_tracker(Return, design, design_idx, False, fidelities[design_idx, :])
                    else:
                        self._design_tracker =  self._design_tracker[ :-1]
                self._last_optimizer_batch_idx = self.optimizer.batch_idx
                return Returns
            mff = mfFunction(f, np.array(self._config["design_bounds"]), np.array(self._config["op_algorithm_config"]["zbounds"]), costmodel=costmodel)
            return mff
        else:
            raise NotImplementedError

    def _define_rl_stuff(self, designs, fidelity):
        # define rl stuff for a design or for a batch of designs to be evaluated
        _batch_size = len(designs)
        if self._config['rl_alg'] == "PPOVecBatch":
            #for PPO vec 
            self._rl_alg = select_rl_alg(self._config['rl_alg'])(self._config['rl_algorithm_config']['algo_params'], self.optimizer.batch_idx, self.env, _batch_size, fidelity)
        else:
            raise NotImplementedError

    def inner_evaluate(self, design_idx):
        #can only accept one (configuration, policy) pair
        if self._config['rl_alg'] == 'PPOVecBatch':
            Return = np.round(self._rl_alg._algorithm_inds[design_idx].evaluate(), 3)
        else:
            raise NotImplementedError
        self.logger.info(
            'end inner (design, policy) pair evaluation for {}-th batch {}-th design: {} after :{}-episodes, the return per step={}, timelimits is :{}'.\
                format(self.optimizer.batch_idx, design_idx, self.current_batch_designs[design_idx], self._design_tracker[self.optimizer.batch_idx][design_idx]['design_episodes'], \
                       Return, self._rl_alg._algorithm_inds[design_idx].time_limit))
        return Return

    def policy_learning(self, design_idx, num_epochs):
        self.logger.info("*******************************************************************************")
        self.logger.info('start policy learning for {}-th batch {}-th design={} fidelity={} with epochs resource={}, timelimits={}'.format(\
            self.optimizer.batch_idx, design_idx, self.current_batch_designs[design_idx], self._rl_alg.fidelity[design_idx], num_epochs, self._rl_alg._algorithm_inds[design_idx].time_limit))
        for _ in range(num_epochs):
            self._single_iteration(design_idx)

    def _single_iteration(self, design_idx):
        #pay attention that before each single iteration, the environment is already set to a specific design
        if self._config['rl_alg'] == 'PPOVecBatch':
            self._rl_alg.single_train_step(design_idx)
            self._episode_counter += 1
            self._design_tracker[self.optimizer.batch_idx][design_idx]['design_episodes'] += 1
            it = self._design_tracker[self.optimizer.batch_idx][design_idx]['design_episodes']
            #os.system("kinit -R")
            if(it%50 == 0):
                _,_reward_ep,_energy,_vel = self._rl_alg._algorithm_inds[design_idx].runner.test()
                self.logger.debug('end policy evaluation for {}-th batch {}-th design:{} after epochs:{} of timelimits:{}'.format(self.optimizer.batch_idx, design_idx,
                                                                                                            self.current_batch_designs[design_idx], it, self._rl_alg._algorithm_inds[design_idx].time_limit))
                self.logger.debug("with ave return={}, ave vel={}, ave energy={}".format(_reward_ep, _vel, _energy))
                self.update_learning_curve(_reward_ep, design_idx)
        else:
            raise NotImplementedError

    def update_design_tracker(self, Return, design, design_idx, last_halving, fidelity = 0):
        '''
        Updates design_tracker after some episodes
        '''
        if 'design' not in self._design_tracker[self.optimizer.batch_idx][design_idx]:
            self._design_tracker[self.optimizer.batch_idx][design_idx]['design'] = design
            self._design_counter += 1
            for i in range(self._config['design_dim']):
                self.writer.add_scalar('Design-vs-time' + '/' + str(i + 1) + '-th-Dimension',
                                       self.env.get_current_design()[i], self._design_counter-1)

        elif design[0] != self._design_tracker[self.optimizer.batch_idx][design_idx]['design'][0]:
            raise ValueError('design assignment error')

        self._design_tracker[self.optimizer.batch_idx][design_idx]['return'].append(Return)
        #todo
        #self.writer.add_scalar('Return-vs-episode', self._design_return[-1], self._episode_counter)

    def update_learning_curve(self, reward_ep, design_idx):

        self.writer.add_scalars('Return-vs-epochs' + '/' + '{}-th-batch'.format(self.optimizer.batch_idx),
                               {'{}-th-design'.format(design_idx): reward_ep}, self._design_tracker[self.optimizer.batch_idx][design_idx]['design_episodes'])

