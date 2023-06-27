from typing import Optional
import gym
import habitat
from habitat.core.dataset import Dataset
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.baseline_registry import baseline_registry
import gzip
import json
import glob
from configs.tunning_config import C
import os
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
    DatasetConfig
)
from habitat import Env


class TaskLess_Env(habitat.RLEnv):
    def __init__(self, env_config: DictConfig, dataset: Dataset = None,logger = None) -> None:
        
        self.env_config = env_config
        self.logger = logger
        # load dataset
        self.split = env_config.habitat.dataset.split
        self.eposide_dir = env_config.habitat.dataset.data_path.format(split = self.split)
        self.scene_dir = env_config.habitat.dataset.scenes_dir
        logger.info("eposide_dir: {}\n".format(self.eposide_dir))
        logger.info("scene_dir: {}\n".format(self.scene_dir))
        content_dir = os.path.join(os.path.dirname(self.eposide_dir), 'content')
        self.file_list = glob.glob(os.path.join(content_dir, '*.json.gz'))
        
        # default to load the first scene
        logger.info(self.file_list[0].split('/')[-1].split('.')[0]) # info simple scene_id of scene
        with habitat.config.read_write(env_config):
                        env_config.habitat.update(
                            {
                                "dataset": DatasetConfig(
                                    data_path = self.file_list[0],
                                    scenes_dir = self.scene_dir
                                )
                            }
                        )
        self.scene_nid = 0
        
        super().__init__(env_config, dataset)
            
            
    def load_new_scene(self, scene_nid = None):
        if scene_nid is None:
            self.scene_nid += 1
            scene_nid = self.scene_nid

        with habitat.config.read_write(self.env_config):
                self.env_config.habitat.update(
                    {
                        "dataset": DatasetConfig(
                            data_path = self.file_list[scene_nid],
                            scenes_dir = self.scene_dir
                        )
                    }
                )
        super().__init__(self.env_config)
        self.logger.info('scene_sid: {}'.format(self.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]))
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
    def reset(self):
        ob = self.habitat_env.reset()
        
    def step(self, action):
        ob, reward, done, info = super().step(action)
        if action == 0:
             self.stop = True
        return ob, reward, done, info

    
    