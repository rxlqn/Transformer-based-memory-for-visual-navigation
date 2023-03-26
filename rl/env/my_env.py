from collections import defaultdict
import sys
sys.path.append('/Extra/lwy/gibson/graduate/')

from rl.env.vec_env import VecEnv
import torch
from typing import Tuple, Union

import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.parallel_env import ParallelNavEnv
import atexit
import multiprocessing
import sys
import traceback
import numpy as np
import os
from gibson2.utils.utils import parse_config
import logging
logging.getLogger().setLevel(logging.WARNING)


class VecGibson(VecEnv):
    def __init__(self) -> None:
        super().__init__()
        num_envs: int
        num_obs: int
        num_privileged_obs: int
        num_actions: int
        max_episode_length: int
        privileged_obs_buf: torch.Tensor
        obs_buf: torch.Tensor 
        rew_buf: torch.Tensor
        reset_buf: torch.Tensor
        episode_length_buf: torch.Tensor # current episode duration
        extras: dict
        device: torch.device
        self.num_obs = 260
        self.num_privileged_obs = None
        self.num_actions = 2
        self.max_episode_length = 500


        config_file_name = '/home/lwy/IGibson2021/iGibson/gibson2/examples/configs/locobot_interactive_nav.yaml'
        env_config = parse_config(config_file_name)
        self.num_envs = 5
        GPU_ID = [1,2] * 5
        self.Env_name = ['Beechwood_1_int','Benevolence_0_int','Ihlen_0_int','Ihlen_1_int','Merom_0_int','Pomaria_0_int','Rs_int','Wainscott_1_int']
        self.Training_Env = self.Env_name[:5] * 2
        self.Testing_Env = self.Env_name[-3:]
        self.core_id = 0
        class load_env(object):
            def __init__(self, num_envs, envs, GPU_ID, i) -> None:
                self.num_envs = num_envs
                self.id = i
                self.envs = envs
                self.GPU_ID = GPU_ID
            def __call__(self, *args, **kwds):
                logging.warning(self.envs[self.id])
                logging.warning(GPU_ID[self.id])
                return iGibsonEnv(config_file = env_config,
                            scene_id = self.envs[self.id],
                            mode = 'headless',
                            action_timestep = 1.0 / 10.0,
                            physics_timestep = 1.0 / 40.0,
                            device_idx = GPU_ID[self.id],
                            automatic_reset = True)
        self.parallel_env = ParallelNavEnv([load_env(self.num_envs, self.Training_Env, GPU_ID, i) for i in range(0, self.num_envs)], blocking=False)      ## env_constructor list contains callable function
        print(self.Training_Env)


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        vec_res = self.parallel_env.step(actions)
        self.obs, rewards, dones, infos = self.process_vec_env(vec_res)
        return self.obs, None, rewards, dones, infos


    def reset(self, env_ids = 'all'):
        '''
            reset state  
            obs includes list of dicts (task_obs rgb and depth)
        '''
        vec_res = self.parallel_env.reset() 
        self.obs = [obs for obs in vec_res]
        return self.obs, None

    def get_observations(self) -> torch.Tensor:
        return self.obs

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return None

    def process_vec_env(self, vec_res):
        '''
            input: vec_res
            output: obs, rewards, dones, infos
        '''
        obs = []
        rewards = []
        dones = []
        infos = defaultdict(list)

        for res in vec_res:
            state, reward, done, info = res
            # if done:
            #     print('done')
            obs.append(state if not done else info['last_observation'])     ## done 后自动reset丢弃第一帧
            rewards.append(reward)
            dones.append(done)
            # infos.append(info)
            info['time_outs'] = True if done and info['episode_length'] == 500 else False
            for key in info:
                infos[key].append(info[key])

        return obs, torch.tensor(np.array(rewards)), torch.tensor(np.array(dones)), infos

    def cal_belief_state(self, state, memory):
        """
        args:
                state: 单张图片
                memory
            
                先计算当前状态的embedding
                更新 memory
                再求出当前的belief_state
        return:
                belief_state
        """
        with torch.no_grad():
            task_obs = state['task_obs'].copy()
            rgb = state['rgb'].copy()
            depth = state['depth'].copy()

            ## T D 只加T
            task_obs = torch.FloatTensor(task_obs).unsqueeze(0).cuda()
            rgb = torch.FloatTensor(rgb).unsqueeze(0).cuda()
            depth = torch.FloatTensor(depth).unsqueeze(0).cuda()
            
            encoder_state, memory = self.encoder_net(rgb,depth,task_obs, 0, memory)
            ## cat predicted angle
            # angle = self.decoder_net(encoder_state) * math.pi
            # encoder_state = torch.cat((encoder_state, angle), -1)
            
        return encoder_state.detach().cpu().numpy(), memory        ## 清除计算图

if __name__ == "__main__":
    env = VecGibson()
    print('ok')