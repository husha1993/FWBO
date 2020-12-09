#default hyperparameters configurations
#set it as default and do not change it, instead change hyperparameters in .json file

from helpers.helpers import deep_update_dict
import copy

default_basic = {'mode': 'train',
                 'force_manual_design': False,
                 'use_cpu_for_rollout': True,
                 'cuda_device': 0,
                 'rl_alg' : 'PPOVecBatch',
                 'op_algorithm_config': dict(
                     plot_interval=10**7,
                     ini_manner='latin_cross',
                     set_mff_target=False,
                     set_mff_inference=True,
                     budget=50,
                     max_num_epochs_per_design=3000, # maximum number of data sampling / training iterates per design
                     seeds=0,
                     ini_seeds=0,
                     GPRS=dict(
                         kernels=dict()
                     ),
                     acq=dict(
                         sampler=dict()
                     ),

                 ),
                 'rl_algorithm_config': dict(
                     num_env=10,
                     concate_design_2_state=False,
                     algo_params=dict(
                         seeds=1,
                         num_steps_per_eval=1000,
                         num_updates_per_epoch=1000, # number of stochastic gradient steps taken per epoch
                         batch_size=256, # number of transitions in the RL batch
                         max_path_length=1000, # max path length for this environment
                         discount=0.99, # RL discount factor
                         reward_scale=1.0,
                         policy_lr=3E-4,
                         qf_lr=3E-4,
                         vf_lr=3E-4,
                         collection_mode='batch',
                         checkpoint_batch=-1, # for PPO: negative value means not saving model
                     ),
                     net_size=200,
                     network_depth=3,
                 ),
                 'env': dict(
                     env_name='HalfCheetah',
                     render=False,
                 ),
                 }


sfbo = {
    'name' : 'Experiment 1: bo',
    'optimizer_name' : 'SfBayesOptimizer',
    'op_algorithm_config' : dict(
                name='SfBayesOptimizer',
    ),
    }

default_basic_sfbo = copy.deepcopy(default_basic)
sfbo = deep_update_dict(sfbo, default_basic_sfbo)

boca_cmfes = {
    'name' : 'Experiment 3: boca_cmfes',
    'optimizer_name' : 'CmfBayesOptimizer_boca',
    'op_algorithm_config' : dict(
        name='CmfBayesOptimizer_boca',
    ),
}
default_basic_boca_cmfes = copy.deepcopy(default_basic)
boca_cmfes = deep_update_dict(boca_cmfes, default_basic_boca_cmfes)

fabolas_cmfes = {
    'name' : 'Experiment 3: fabolas_cmfes',
    'optimizer_name' : 'CmfBayesOptimizer_fabolas',
    'op_algorithm_config' : dict(
        name='CmfBayesOptimizer_fabolas',
    ),
}
default_basic_fabolas_cmfes = copy.deepcopy(default_basic)
fabolas_cmfes = deep_update_dict(fabolas_cmfes, default_basic_fabolas_cmfes)

fidelitywarping_cmfes = {
    'name' : 'Experiment 3: fidelitywarping_cmfes',
    'optimizer_name' : 'CmfBayesOptimizer_fidelitywarping',
    'op_algorithm_config' : dict(
        name='CmfBayesOptimizer_fidelitywarping',
    ),
}
default_basic_fidelitywarping_cmfes = copy.deepcopy(default_basic)
fidelitywarping_cmfes = deep_update_dict(fidelitywarping_cmfes, default_basic_fidelitywarping_cmfes)

random = {
    'name' : 'Experiment 3: random',
    'optimizer_name' : 'randomOptimizer',
    'op_algorithm_config' : dict(
        name='randomOptimizer',
    ),
}
default_basic_random = copy.deepcopy(default_basic)
random = deep_update_dict(random, default_basic_random)

cmaes = {
    'name' : 'Experiment 3: cmaes',
    'optimizer_name' : 'cmaesOptimizer',
    'op_algorithm_config' : dict(
        name='cmaesOptimizer',
    ),
}
default_basic_cmaes = copy.deepcopy(default_basic)
cmaes = deep_update_dict(cmaes, default_basic_cmaes)


hpcbbo = {
    'name' : 'Experiment 3: hpcbbo',
    'optimizer_name' : 'hpcbboOptimizer',
    'op_algorithm_config' : dict(
        name='hpcbboOptimizer',
    ),
}
default_basic_hpcbbo = copy.deepcopy(default_basic)
hpcbbo = deep_update_dict(hpcbbo, default_basic_hpcbbo)


config_dict = {
    'sfbo': sfbo,
    'boca': boca_cmfes,
    'fabolas': fabolas_cmfes,
    'fidelitywarping': fidelitywarping_cmfes,

    'random': random,
    'cmaes': cmaes,
    'hpcbbo': hpcbbo
    }

if __name__ == "__main__":
    sfbo = config_dict['sfbo']
    print(sfbo)
