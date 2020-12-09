import time

import numpy as np

from .PPO_vec import PPO_vec



class PPOVecBatch(object):

    def __init__(self, config, batch_index, vec_env, batch_size, fidelity):
        self.config = config
        self.batch_size = batch_size
        self._algorithm_inds = dict()
        if not isinstance(fidelity, np.ndarray):
            # when fidelity is not a numpy array
            self.fidelity = np.ones((batch_size, 1)) * fidelity
        else:
            self.fidelity = fidelity
        if type(fidelity) != int:
            if self.fidelity.shape[0] != self.batch_size:
                raise ValueError('the num of fidelities and the batch_size should be the same')

        for i in range(self.batch_size):
            # to avoid the naming conflicts when the n_init of bayesian optimization is more than 1
            exp_id_prefix = 'ini_' if batch_index == 0 else ''
            exp_id_prefix += time.strftime("%d-%H-%M_", time.localtime())

            fname = '_'.join([str(i) for i in list(self.fidelity[i])])
            self._algorithm_inds[i] = PPO_vec(
                vec_env = vec_env,
                exp_id = exp_id_prefix+"f="+ fname + "_batchIdx=" + str(batch_index)+ "_" + str(self.batch_size * batch_index+i),
                sample_size = config['sample_size'] if 'sample_size' in config.keys() else 4096,
                epoch_size = config['epoch_size'] if  'epoch_size' in config.keys() else 10,
                batch_size = config['batch_size'] if 'batch_size' in config.keys() else 256,
                clip_threshold = config['clip_threshold'] if 'clip_threshold' in config.keys() else 0.2,
                save_dir = config['save_dir'],
                lam = config['lam'] if 'lam' in config.keys() else 0.95,
                actor_lr = config['actor_lr'] if 'actor_lr' in config.keys() else 3e-4,
                critic_lr = config['critic_lr'] if 'critic_lr' in config.keys() else 3e-4,
                
                v_max = config['v_max'],
                v_min = config['v_min'],
                time_limit = config['time_limit'],
                checkpoint_batch = config['checkpoint_batch'],
                seed = config["seeds"]
            )

    def single_train_step(self, design_index):
        self._algorithm_inds[design_index]._try_to_train()

    def evaluate(self,design_index):
        return self._algorithm_inds[design_index].evaluate()

    def changeTimeLimit(self, timeLimit):
        for i in range(len(self._algorithm_inds)):
            self._algorithm_inds[i].time_limit = timeLimit

    def getModelPath(self, design_index):
        #get save dir of logs
        return self._algorithm_inds[design_index].model_path

    def loadModel(self, design_index, model_path):
        #load pretrained model
        self._algorithm_inds[design_index].loadModel(model_path)



