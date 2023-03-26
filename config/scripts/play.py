# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import sys
sys.path.append('/Extra/lwy/gibson/graduate/')
from rl.env.my_env import VecGibson
import numpy as np
import torch


import argparse

# from simple_agent import RandomAgent, ForwardOnlyAgent
# from rl_agent import SACAgent
from gibson2.utils.utils import parse_config
from gibson2.challenge.challenge import Challenge
from gibson2.envs.igibson_env import iGibsonEnv
import os
import torch
import datetime
from rl.utils.log_utils import ini_logger
from rl.utils.logging_engine import logger
from rl.modules import ActorCritic, ActorCriticRecurrent
from config.utils import task_registry
from rl.env.my_env import VecGibson
from config.envs.Gibson.train_config import GibsonCfgPPO

import datetime
def main():

    log_file_name = f"test_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.log"
    ini_logger(log_file_name, level='info')
    model_path = '/Extra/lwy/gibson/graduate/logs/igibson_all/Feb17_02-43-54_/model_3500.pt'
    # model_path = './transformer_waypoints/11_22/model/SAC_smtI_32_waypoints_11_22_std800'
    logger.info(f"Start to run {model_path}")
    env = VecGibson()

    # load policy
    GibsonCfgPPO.runner.resume = True
    GibsonCfgPPO.runner.log_root = '/Extra/lwy/gibson/graduate/logs/igibson_all/'
    GibsonCfgPPO.runner.load_run = 'Feb17_02-43-54_'
    GibsonCfgPPO.runner.checkpoint = '3500'
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name='gibson', train_cfg=GibsonCfgPPO)
    policy = ppo_runner.get_inference_policy(device='cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test(policy, ppo_runner, 0)

def test(agent, runner, gpu):
    config_file = '/home/lwy/IGibson2021/iGibson/gibson2/examples/configs/locobot_interactive_nav.yaml'
    split = 'test'  ## train
    episode_dir = '/home/lwy/IGibson2021/iGibson/gibson2/data/episodes_data/interactive_nav'

    eval_episodes_per_scene = os.environ.get(
        'EVAL_EPISODES_PER_SCENE', 100)

    env_config = parse_config(config_file)
    task = env_config['task']

    logger.info(f'{task},{split}')
    if task == 'interactive_nav_random':
        metrics = {key: 0.0 for key in [
            'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}

    elif task == 'social_nav_random':
        metrics = {key: 0.0 for key in [
            'success', 'stl', 'psc', 'episode_return']}
    else:
        assert False, 'unknown task: {}'.format(task)

    num_episodes_per_scene = eval_episodes_per_scene
    split_dir = os.path.join(episode_dir, split)
    assert os.path.isdir(split_dir)
    num_scenes = len(os.listdir(split_dir))
    assert num_scenes > 0
    total_num_episodes = num_scenes * num_episodes_per_scene

    idx = 0
    for json_file in os.listdir(split_dir):
        scene_id = json_file.split('.')[0]
        json_file = os.path.join(split_dir, json_file)
        logger.info(json_file)
        env_config['scene_id'] = scene_id
        env_config['load_scene_episode_config'] = True
        env_config['scene_episode_config_name'] = json_file
        env = iGibsonEnv(config_file=env_config,
                            mode='headless',
                            action_timestep=1.0 / 10.0,
                            physics_timestep=1.0 / 40.0,
                            device_idx=gpu)
        scene_metrics = {key: 0.0 for key in [
            'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        for _ in range(num_episodes_per_scene):
            idx += 1
            state = env.reset()
            # memory = torch.FloatTensor([]).cuda()
            # belief_state, memory = agent.cal_belief_state(state, memory)
            episode_return = 0.0
            while True:
                # action = env.action_space.sample()
                action = runner.alg.act([state], [state])   
                state, reward, done, info = env.step(action[0])
                # belief_state, memory = agent.cal_belief_state(state, memory)

                episode_return += reward
                if done:
                    logger.info(f'Episode: {idx}/{total_num_episodes},  return :{episode_return}')
                    break

            metrics['episode_return'] += episode_return
            scene_metrics['episode_return'] += episode_return
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]
                    scene_metrics[key] += info[key]

        for key in metrics:
            scene_metrics[key] /= num_episodes_per_scene
            logger.info('Avg {}: {}'.format(key, scene_metrics[key]))
        
        env.close()

    for key in metrics:
        metrics[key] /= total_num_episodes
        logger.info('Avg {}: {}'.format(key, metrics[key]))
    return metrics['episode_return']

if __name__ == "__main__":
    main()
