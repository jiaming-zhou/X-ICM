from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, euler_to_quaternion, CAMERAS
import torch
from utils import convert_to_euler

class CustomModelAgent(Agent):
    def __init__(self, task_name, seed=0, action_chunk_length=1):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.front_rgb_path=None
        self.seed=seed
        self.action_chunk_length=action_chunk_length

    def _inference(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        lang_goal = kwargs['lang_goal']


        ### TODO #1: construct your custom model input here
        ### you can read the information your model need from the "obs"
        ### example: we show example input for Pi0 model here
        front_image = np.transpose((obs['front_rgb'][0,0].cpu().numpy()),(1,2,0)).astype(np.uint8)
        wrist_image = np.transpose((obs['wrist_rgb'][0,0].cpu().numpy()),(1,2,0)).astype(np.uint8)
        overhead_image = np.transpose((obs['overhead_rgb'][0,0].cpu().numpy()),(1,2,0)).astype(np.uint8)
        gripper_pose=obs['gripper_pose'][0,0].cpu().numpy()
        gripper_pose_euler=np.concatenate((gripper_pose[:3], convert_to_euler(gripper_pose[3:])))
        state = np.concatenate((gripper_pose_euler, obs['gripper_open'][0,0].cpu().numpy()))
        example = { "observation/front_image": front_image,
                    "observation/wrist_image": wrist_image,
                    "observation/overhead_image": overhead_image,
                    "observation/state": state,
                    "prompt": lang_goal,
                }


        ### TODO #2: call your custom model here
        ### example: we show inference of Pi0 model here
        action_chunk=self.policy.infer(example)['actions']
    

        ### TODO #3: postprocess the action_chunk here
        ### convert the action_chunk to the format of the RLBench's action space 
        ### [3-dim translation, 4-dim quaternion, 1-dim gripper_open, 1-dim collision (default to 1)]
        ### example: we show postprocess of Pi0 model here
        if len(np.array(action_chunk).shape) == 1:
            action_chunk = [action_chunk]

        if len(action_chunk)>self.action_chunk_length:
            action_chunk=action_chunk[:self.action_chunk_length]
        
        if len(np.array(action_chunk).shape) == 1:
            action_chunk = [action_chunk]

        executable_actions = []
        for action in action_chunk:
            trans_indicies = np.array(action[:3])
            rot_indicies = np.array(action[3:6])
            is_gripper_open = 1 if action[6]>=0.5 else 0

            executable_action = np.concatenate(
                [
                    np.array(trans_indicies),
                    euler_to_quaternion(rot_indicies, degree=False),  
                    [is_gripper_open],
                    [1],
                ]
            )

            executable_actions.append(executable_action)

        return executable_actions
        

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        # inference
        if len(self.actions) == 0:
            self.actions = self._inference(observation, step, **kwargs)
            
        continuous_action = self.actions.pop(0)

        self.step += 1
        
        # copy_obs = {k: v.cpu() for k, v in observation.items()}
        copy_obs={}
        for k, v in observation.items():
            if k=='lang_goal':
                copy_obs[k]=v
            else:
                copy_obs[k]=v.cpu()
        
        return ActResult(continuous_action,
                         observation_elements=copy_obs,
                         info=None)
    
    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str, components):

        ### TODO: load your custom model here
        self.policy = None
        # self.policy = your_model_load_func(your_checkpoint_dir)

        return


    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return