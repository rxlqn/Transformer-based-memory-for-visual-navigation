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


if __name__ == "__main__":
    config_file_name = '/home/lwy/IGibson2021/iGibson/gibson2/examples/configs/locobot_interactive_nav.yaml'
    env_config = parse_config(config_file_name)
    GPU_ID = [0,0,0,1,1,1,2,2]
    Env = ['Beechwood_1_int','Benevolence_0_int','Ihlen_0_int','Ihlen_1_int','Merom_0_int','Pomaria_0_int','Rs_int','Wainscott_1_int']
    Training_Env = Env[:5]
    Testing_Env = Env[-3:]
    core_id = 0
    num_env = 2
    def load_env():
        global core_id
        core_id = core_id + 1
        return iGibsonEnv(config_file = env_config,
                        scene_id = Training_Env[core_id],
                        mode = 'headless',
                        action_timestep = 1.0 / 10.0,
                        physics_timestep = 1.0 / 40.0,
                        device_idx = GPU_ID[core_id],
                        automatic_reset = True)

    parallel_env = ParallelNavEnv([load_env] * num_env, blocking=False)


    from time import time
    for episode in range(10):
        start = time()
        print("episode {}".format(episode))
        parallel_env.reset()
        for i in range(600):
            res = parallel_env.step([[0.5, 0.5] for _ in range(2)])
            state, reward, done, _ = res[0]
            if done:    ## 设置了自动重启，if done，从info['last_observation'] 中取最后的数据，此时返回的state为reset后获得的状态
                print("Episode finished after {} timesteps".format(i + 1))
                # break
        print("{} elapsed".format(time() - start))