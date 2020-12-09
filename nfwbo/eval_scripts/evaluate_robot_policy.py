import os, inspect, socket
import json
import argparse
import subprocess as sp
import time
import torch
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

import configs.default_configs as default_cfgs
from helpers.helpers import set_random_seeds, save_args2json, create_logging, deep_update_dict, printProgressBar
from envs.pybulletevo.evoenvs import HalfCheetahUrdfEnv, AntEnv
from algorithm.model import load_model
from RL.env.clEnv import CLEnv


def create_save_path(args, sublist, mode='train', extra=''):
    host_name = socket.gethostname()
    current_dtime = time.strftime("%m-%d", time.localtime())
    current_htime = time.strftime("%H-%M-%S", time.localtime())

    nfa = ''  # name from args
    for i in sublist:
        if type(args) == dict:
            nfa += '-' + i + '=' + str(args[i])
        else:
            nfa += '-' + i + '=' + str(getattr(args, i))

    save_path = './{0}_dir/{1}/{2}/{3}'.format(
        mode,
        current_dtime + '_video_icra2021',
        args['config'] if type(args) == dict else args.config,
        host_name
        + '-st=' + current_htime
        + nfa
        + '-'
        + extra)

    return save_path


def select_environment(env_name):
    if env_name == "HalfCheetahUrdfEnv":
        return HalfCheetahUrdfEnv
    elif env_name == 'AntEnv':
        return AntEnv
    else:
        raise ValueError("Environment class not found.")


vis_checklist = ['steps', 'rewards', 'ave_rewards']


def add_text(draw, vis_info, ite_step, total_steps):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    font = ImageFont.truetype(os.path.join(currentdir, 'sans-serif.ttf'))
    line = 0
    pb = printProgressBar(ite_step, total_steps)
    draw.text((10, 10 * (line + 1)), pb, fill=(0, 0, 0), font=font)
    line += 1
    for _ in range(len(vis_checklist)):
        vis = vis_checklist[_]
        if vis in vis_info:
            if vis == 'v(m/s)':
                draw.text((10, 10 * (line + 1)), vis + "=[vx, vy, vz]", fill=(0, 0, 0), font=font)
                line += 1
            if vis == 'pose(m)':
                draw.text((10, 10 * (line + 1)), vis + "=[x, y, z]", fill=(0, 0, 0), font=font)
                line += 1
            draw.text((10, 10 * (line + 1)), vis + "={}".format(str(vis_info[vis])), fill=(0, 0, 0), font=font)
            line += 1


def main_eval(args):
    eval_c_file = os.path.join(args.exp_dir, 'config.json')
    eval_a_file = os.path.join(args.exp_dir, 'args.json')
    eval_p_file = os.path.join(args.exp_dir, 'model', args.model_dir, 'checkpoint_' + args.model_name + '.tar')
    # Read args from saved args file
    with open(eval_a_file) as f:
        eval_args = json.load(f)

    # Read configs from saved config file
    default_config = default_cfgs.config_dict[eval_args['m']]
    with open(eval_c_file) as f:
        eval_config = json.load(f)
        config = deep_update_dict(eval_config, default_config)

    set_random_seeds(eval_args['seeds'])
    file_str = create_save_path(eval_args, ['seeds'], 'eval', 'exp_dir=' + args.method + args.exp_dir.replace('/','-'))
    config['data_folder_experiment'] = file_str
    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    # configure and create logging
    logger = create_logging('output.log', file_str, 'w', args.debug)
    logger.info('Current config content is :{}'.format(config))
    writer = SummaryWriter(log_dir=file_str)

    env = CLEnv(select_environment(config['env']['env_name'])(config, 6))
    env.set_new_design(args.design)

    # Store args
    save_args2json('args.json', file_str, args)

    total_steps = config["op_algorithm_config"]["time_limits"] + 49
    env.set_task_t(total_steps)
    s_norm, actor, critic = load_model(eval_p_file)

    acc_step = 0
    acc_rwd = 0
    for i in range(args.ei):
        if args.save_video:
            video_path = os.path.join(file_str, 'videos', str(i) + '-th-iteration')
            os.makedirs(video_path, exist_ok=True)

        obs, info = env.reset(return_info=True)
        obs = torch.Tensor(obs).float()

        done = False
        ite_step = 0
        ite_rwd = 0
        vis_info = {}

        while True:
            vis_info['design'] = env.get_current_design()
            vis_info['steps'] = ite_step
            for vis in vis_checklist:
                if vis in info:
                    vis_info[vis] = info[vis]

            if args.save_video:
                rgb = env.render('rgb_array')
                rgb = Image.fromarray(rgb)
                traj = Image.new('RGB', (rgb.size[0] + 130, rgb.size[1]), (255, 255, 255))
                if 'pose(m)' in info:
                    x, y, z = info['pose(m)']
                    x = 8*x
                    y = 8*y

                    vis_info.setdefault("traj", []).append((x + 10, traj.size[1] - (y + traj.size[1] / 3)))
                elif 'pos' in info:
                    x, y, z = info['pos']
                    x = 8*x
                    y = 8*y

                    vis_info.setdefault("traj", []).append((x + 10, traj.size[1] - (y + traj.size[1] / 3)))
                all = Image.new('RGB', (traj.size[0] + rgb.size[0], rgb.size[1]))
                draw_rgb = ImageDraw.Draw(rgb)
                draw_traj = ImageDraw.Draw(traj)
                add_text(draw_traj, vis_info, ite_step, total_steps)
                if 'traj' in vis_info:
                    draw_traj.point(vis_info['traj'], fill='red')
                all.paste(rgb, (0, 0))
                all.paste(traj, (rgb.size[0], 0))
                all.save(os.path.join(video_path, '{:05d}.png'.format(ite_step)))
            obst = torch.FloatTensor(obs)
            s_norm.record(obst)
            obst_norm = s_norm(obst)

            with torch.no_grad():
                ac = actor.act_deterministic(obst_norm)
                ac = ac.cpu().numpy()
            if done:
                break

            obs, rwd, done, info = env.step(ac)

            time.sleep(0.0165)

            ite_step += 1
            ite_rwd += rwd
            info['rewards'] = ite_rwd
            info['ave_rewards'] = ite_rwd/ite_step
            acc_rwd += rwd
            acc_step += 1
        logger.info('return of {}-th iteration is {}, and num of steps is {}, return per step={}'.format(i, ite_rwd, ite_step, ite_rwd/250))

        if args.save_video:
            sp.call(['ffmpeg', '-loglevel', 'panic', '-r', '60', '-f', 'image2', '-i', os.path.join(video_path, '%05d.png'), '-vcodec', 'libx264',
                 '-pix_fmt', 'yuv420p', os.path.join(video_path, str(i) + '-th-ite_rwd=' + str(int(ite_rwd)) + '.mp4')])

    avg_step = acc_step/args.ei
    avg_rwd = acc_rwd/args.ei
    logger.info('average return of {} iterations is {}, and num of steps is {}, average return per step={}'.format(args.ei, avg_rwd, avg_step, avg_rwd/avg_step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('eval and make videos')
    #saved experiments directory
    parser.add_argument('--exp_dir', type=str, default=None, required=True)
    parser.add_argument('--model_dir', type=str, default=None, required=True)
    parser.add_argument('--model_name', type=str, default='240')

    #eval settings
    parser.add_argument('--ei', type=int, default=10)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')

    args = parser.parse_args()
    main_eval(args)
