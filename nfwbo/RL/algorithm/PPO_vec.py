import time
import os
import random
import logging
_log = logging.getLogger(__name__)

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import numpy as np
np.set_printoptions(precision=16)
torch.set_printoptions(precision=16)

from .model import *
from .runner import GAERunner
from .data_tools import PPO_Dataset
from helpers.helpers import set_random_seeds as set_seed


def calc_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad == True:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1./2)
    return total_norm


class PPO_vec(object):

    def __init__(self,vec_env, exp_id,
        save_dir="./experiments",
        sample_size=4096,
        epoch_size=10,
        batch_size=256,
        checkpoint_batch=-1,
        test_batch=20,
        gamma=0.99,
        lam=0.95,
        clip_threshold=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        max_grad_norm=0.5,
        v_max = 1,
        v_min = 0,
        time_limit = 1000,
        use_gpu_model=False,
        CL = False,
        seed = 0):

        #log parameters
        self.save_dir = save_dir
        self.checkpoint_batch = checkpoint_batch
        self.test_batch = test_batch
        self.exp_id = exp_id
        self.initLog()

        #rl parameters
        self.sample_size = sample_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_threshold = clip_threshold
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_grad_norm = max_grad_norm
        self.v_max = v_max
        self.v_min = v_min
        self.time_limit = time_limit
        self.use_gpu_model = use_gpu_model
        self.CL = CL
        self.seed = seed

        #network settings
        self.s_norm = Normalizer(vec_env.observation_space.shape[0])
        set_seed(self.seed)
        self.actor = Actor(vec_env.observation_space.shape[0], vec_env.action_space.shape[0], vec_env.action_space, hidden=[128, 64])
        self.critic = Critic(vec_env.observation_space.shape[0], self.v_min/(1-self.gamma), self.v_max/(1-self.gamma), hidden =[128, 64])

        #env setting
        self.vec_env = vec_env
        self.runner = GAERunner(self.vec_env, self.s_norm, self.actor, self.critic, self.sample_size, self.gamma, self.lam, self.v_max, self.v_min,
            use_gpu_model = self.use_gpu_model)


        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), self.critic_lr)

        if use_gpu_model:
            self.T = lambda x: torch.cuda.FloatTensor(x)
        else:
            self.T = lambda x: torch.FloatTensor(x)

        #idx of iter
        self.it = 0
        self.continuous = False
        self.nocontinuous = False
        self.reuse = False

        self.rwd_list=[]
        self.vel_list=[]
        self.energy_list = []
        self.continous_rwd_list=[]
        self.continous_steps_list=[]


    def _try_to_train(self):
        '''
        train one iter
        '''
        if(self.it == 0):
            set_seed(self.seed)

        t = self.time_limit
        self.vec_env.set_task_t(self.time_limit)

        data = self.runner.run()
        dataset = PPO_Dataset(data)

        atarg = dataset.advantage
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5)

        adv_clip_rate = np.mean(np.abs(atarg) > 4)
        adv_max = np.max(atarg)
        adv_min = np.min(atarg)
        val_min = self.v_min;
        val_max = self.v_max / (1-self.gamma);
        vtarg = dataset.vtarget
        vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
        vtd_max = np.max(vtarg)
        vtd_min = np.min(vtarg)

        atarg = np.clip(atarg, -4, 4)
        vtarg = np.clip(vtarg, val_min, val_max)

        dataset.advantage = atarg
        dataset.vtarget = vtarg

        # logging interested variables
        N = np.clip(data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
        avg_rwd = data["rwds"].sum()/N
        avg_step = data["samples"]/N
        rwd_per_step = avg_rwd / avg_step

        fail_rate = sum(data["fails"])/N
        self.total_sample += data["samples"]


        if (self.it % self.test_batch == 0):
            _log.debug('*****start to evaluate by runner at iter={}'.format(self.it))
            test_step, test_rwd, test_energy, test_vel = self.runner.test()
            _log.debug('*****end evaluating by runner at iter={}'.format(self.it))
            test_avg_rwd = test_rwd/self.vec_env.task_t# record average rewards
            #if(self.continuous == False or self.nocontinuous == True):
            self.rwd_list.append(test_rwd)
            self.energy_list.append(test_energy)
            self.vel_list.append(test_vel)

            if(self.continuous == True):
                self.continous_rwd_list.append(test_rwd)
                self.continous_steps_list.append(test_step)


            print("\n===== iter %d ====="% self.it)
            print("test_task_t       = %f" % self.vec_env.task_t)
            print("test_avg_rwd       = %f" % test_avg_rwd)
            print("test_rwd      = %f" % test_rwd)
            print("test_step     = %f" % test_step)

        # start training
        pol_loss_avg    = 0
        pol_surr_avg    = 0
        pol_abound_avg  = 0
        vf_loss_avg     = 0
        clip_rate_avg   = 0

        actor_grad_avg  = 0
        critic_grad_avg = 0



        gradient_norm = 0
        for epoch in range(self.epoch_size):
        #print("iter %d, epoch %d" % (it, epoch))
            for bit, batch in enumerate(dataset.batch_sample(self.batch_size)):
                # prepare batch data
                ob, ac, atarg, tdlamret, log_p_old = batch
                ob = self.T(ob)
                ac = self.T(ac)
                atarg = self.T(atarg)
                tdlamret = self.T(tdlamret).view(-1, 1)
                log_p_old = self.T(log_p_old)

                # clean optimizer cache
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                # calculate new log_pact
                ob_normed = self.s_norm(ob)
                m = self.actor.act_distribution(ob_normed)
                vpred = self.critic(ob_normed)
                log_pact = m.log_prob(ac)
                if log_pact.dim() == 2:
                    log_pact = log_pact.sum(dim=1)

                # PPO object, clip advantage object
                ratio = torch.exp(log_pact - log_p_old)
                surr1 = ratio * atarg
                surr2 = torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * atarg
                pol_surr = -torch.mean(torch.min(surr1, surr2))
                pol_loss = pol_surr
                pol_loss_avg += pol_loss.item()


                # critic vpred loss
                vf_criteria = nn.MSELoss()
                vf_loss = vf_criteria(vpred, tdlamret) / (self.critic.v_std**2) # trick: normalize v loss


                vf_loss_avg += vf_loss.item()

                if (not np.isfinite(pol_loss.item())):
                    print("pol_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                if (not np.isfinite(vf_loss.item())):
                    print("vf_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                pol_loss.backward()
                vf_loss.backward()


                gradient_norm += calc_grad_norm(self.critic)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)



                self.actor_optim.step()
                self.critic_optim.step()


        batch_num = (self.sample_size // self.batch_size)
        pol_loss_avg    /= batch_num
        vf_loss_avg     /= batch_num

        # save checkpoint
        if (self.checkpoint_batch > 0 and self.it % self.checkpoint_batch == 0 and self.it > 0):
            _log.debug("save check point to:{}/{}/checkpoint_{}".format(self.save_dir, self.exp_id, self.it))
            self.actor.cpu()
            self.critic.cpu()
            self.s_norm.cpu()
            data = {"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "s_norm": self.s_norm.state_dict()}
            if self.use_gpu_model:
                self.actor.cuda()
                self.critic.cuda()
                self.s_norm.cuda()
            if(t>=100):
                if(self.continuous == True):
                    torch.save(data, "%s/%s/transfer_checkpoint_%d.tar" % (self.save_dir, self.exp_id, int(t)))
                elif(self.continuous == False and self.reuse == True):
                    torch.save(data, "%s/%s/continuous_checkpoint_%d.tar" % (self.save_dir, self.exp_id, int(t)))

            torch.save(data, "%s/%s/checkpoint_%d.tar" % (self.save_dir, self.exp_id, self.it))
        self.it +=1



    def setExpid(self,idx):
        '''
        set id of the experiment(idx of batch in BayesianOpt)
        '''
        self.exp_id = idx

    def initLog(self):
        '''
        initialize tensorboard log settings
        '''
        #self.writer = SummaryWriter("%s/%s" % (self.save_dir, self.exp_id))
        if(not os.path.exists(self.save_dir + "/" + str(self.exp_id))):
            os.makedirs(self.save_dir + "/" + str(self.exp_id))
        self.model_path = self.save_dir + "/" + str(self.exp_id)
        self.total_sample = 0
        self.train_sample = 0

    def killLog(self):
        #self.writer.close()
        self.total_sample = 0
        self.train_sample = 0
        self.it = 0

    def learn(self, num_iter):
        '''
        train policy for num_iter iterations
        '''
        set_seed(self.seed)
        for i in range(num_iter):
            self._try_to_train()


    def evaluate(self):
        '''
        evaluate trained policy
        '''
        rwd_list = self.rwd_list[-5:]
        rwd_list.sort()
        rewards = rwd_list[-2:]
        _log.debug('***start evaluating the trained policy with self.rwd_list[-5:]={}, rewards={}, self.time_limit={}'.format(rwd_list, rewards, self.time_limit))
        rewards = np.array(rewards).mean()/self.time_limit
        return rewards

    def loadModel(self, ckpt):
        s_norm, actor, critic = load_model(ckpt)
        self.s_norm = s_norm
        self.actor = actor
        self.critic = critic
        self.runner =  GAERunner(self.vec_env, self.s_norm, self.actor, self.critic, self.sample_size, self.gamma, self.lam, self.v_max, self.v_min,
            use_gpu_model = self.use_gpu_model)
        self.actor_optim = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), self.critic_lr)

    def testf(self, ckpt, task_length):
        # given pretrained  model and task length, test its performance
        print(ckpt)
        self.loadModel(ckpt)
        self.vec_env.set_task_t(task_length)
        steps, rwd, _, _ = self.runner.test()
        print(task_length)
        print(steps)
        return  rwd/task_length


