import os
import json
import argparse

from tensorboardX import SummaryWriter

from design_optimize_alg.DesignOptimize import DesignOptimize
import configs.default_configs as default_cfgs
from helpers.helpers import create_save_path, set_random_seeds, save_args2json, create_logging, deep_update_dict
from helpers.helpers import compute_resources_of_optimizer


def check_config(config, args):
    # a set of rules to make sure that the config file is logical
    if config['design_dim'] != len(config['design_bounds'][0]) or config['design_dim'] != len(config['design_bounds'][1]):
        raise ValueError('must set the design bounds of the same dimension as design_dim')
    if not args.config.startswith('debug') and not args.config.startswith(args.m):
        raise ValueError('must set the method the same as the config json file')


def main(args):
    # Read config from config files
    default_config = default_cfgs.config_dict[args.m]
    if args.config:
        new_config_file = os.path.join('./nfwbo/configs', args.robot, args.config + '.json')
        with open(new_config_file) as f:
            new_config = json.load(f)
        config = deep_update_dict(new_config, default_config)
    else:
        config = default_config
    # check possible bugs in config settings
    check_config(config, args)

    set_random_seeds(args.seeds)

    # change config file by arguments from args
    config["op_algorithm_config"]["seeds"] = args.seeds

    if args.dev:
        config["op_algorithm_config"]["max_num_epochs_per_design"] = 1
        config["op_algorithm_config"]["time_limits"] = 1
        config["op_algorithm_config"]["x_ninit"] = 1
        config["op_algorithm_config"]["z_ninit"] = 1
        config["op_algorithm_config"]["optniter"] = 1
        config["op_algorithm_config"]["optninit"] = 1
        config["op_algorithm_config"]["batch_size"] = 2
        config['rl_algorithm_config']["algo_params"]["checkpoint_batch"] = -1
    config['force_manual_design'] = args.h

    file_str = create_save_path(args, ['seeds']) + '_' + args.extra
    if args.h:
        file_str = file_str + '_human_design'
        config['rl_algorithm_config']["algo_params"]["checkpoint_batch"] = 240
        config['op_algorithm_config']["set_mff_inference"] = False

    config['data_folder_experiment'] = file_str
    plt_dir = os.path.join(file_str, 'plt')
    config['op_algorithm_config']['plt_dir'] = plt_dir
    model_path = file_str + "/model"
    config["rl_algorithm_config"]["algo_params"]["save_dir"] = model_path

    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)
    #create model's folder
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    # Store config
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
        fd.write(json.dumps(config, indent=2))
    # Store args
    save_args2json('args.json', file_str, args)

    # configure and create logging
    logger = create_logging('output.log', file_str, 'w', args.debug)
    logger.info('Current config content is :{}'.format(config))
    writer = SummaryWriter(log_dir=file_str)

    do = DesignOptimize(logger, writer, config)
    do.optimize()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Miscellaneous
    parser.add_argument('--seeds', type=int, default=1, help='set random seed for numpy and torch')
    parser.add_argument('--debug', default=True, action='store_true')
    parser.add_argument('--dev', default=False, action='store_true', help='fast developing mode for debug')
    parser.add_argument('--h', default=False, action='store_true', help='force human design')
    parser.add_argument('--extra', type=str, default='_', help='extra name to specify purpose of the experiment')

    # Method
    parser.add_argument('--m', type=str, default='sfbo', help='which method to use')
    parser.add_argument('--config', type=str, default=None, required=True)

    # Robot-Sim-Task setting
    parser.add_argument('--robot', type=str, default='cheetah')

    args = parser.parse_args()
    main(args)
