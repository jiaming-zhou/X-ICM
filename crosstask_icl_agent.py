from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from vllm import LLM, SamplingParams 
from qwen_vl_utils import process_vision_info


class CrossTaskICLAgent(Agent):
    def __init__(self, task_name, demo_num_per_icl=10, seed=0, ranking_method="lang_vis.out"):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.demo_num_per_icl = demo_num_per_icl
        self.front_rgb_path=None
        self.seed=seed
        self.ranking_method=ranking_method

        self.SYSTEM_PROMPT = "You are a Franka Panda robot with a parallel gripper. We provide you with some demos from some seen tasks, in the format of [task_instruction, observation]>[ 7-dim action_1, 7-dim action_2, ..., 7-dim action_N ]. Then you will receive an unseen task instruction with a new observation, and you need to output a list of 7-dim actions that match the trends in the demos. Do not output anything else."


    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        lang_goal = kwargs['lang_goal']

        front_rgb_img = obs['front_rgb']
        front_rgb_img=front_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        front_rgb_img = np.clip((front_rgb_img).astype(np.uint8), 0, 255)

        front_rgb_img = Image.fromarray(front_rgb_img)
        front_rgb_dir = os.path.join(self.savedir, 'rgb_dir', 'front', str(self.episode_id))
        os.makedirs(front_rgb_dir, exist_ok=True)
        front_rgb_img.save(os.path.join(front_rgb_dir, 'rgb.png'))
        self.front_rgb_path=os.path.join(front_rgb_dir, 'rgb.png')

        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)

            rgb_dict[camera] = rgb_img

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy() 

            mask_dict[camera] = mask

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        if len(self.actions) == 0:
            user_prompt = self.handler.get_user_prompt_ranking(mask_dict, mask_id_to_sim_name, point_cloud_dict, custom_num_demos=self.demo_num_per_icl, taskname=lang_goal, image_path=self.front_rgb_path, seed=self.seed, ranking_metric=self.ranking_method)   

            print(self.SYSTEM_PROMPT) 

            print()

            print(user_prompt)

            messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]


            ########################### vllm local deploy #####################################
            prompt = self.components["processor"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
                
            llm_inputs = {
                "prompt": prompt
            }
            

            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=256,
                stop_token_ids=[],
            )

            outputs = self.components["llm"].generate([llm_inputs], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text

            print(f"Prediction:", output_text)
            return output_text
    
    def re_match(self, text):
        pattern = r'\[([^\[\]]+\d[^\[\]]*)\]'
        matches = re.findall(pattern, text)
        
        valid_lists = []
        for match in matches:
            items = [int(x.strip()) for x in match.split(',')]
            if len(items) == 7:
                valid_lists.append(items)
        return valid_lists

    def _postprocess(self, output_text):
        try:
            actions = self.re_match(str(output_text))
            print("parsed actions: ", actions)
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions. Falling back to default.')
        
        if len(np.array(actions).shape) == 1:
            actions = [actions]

        output = []
        for action in actions:
            if len(action) != 7:
                print("error:::", actions)
                if len(action)==6:
                    action.append(1)
                else:
                    action = [57, 49, 87, 0, 39, 0, 1]
            trans_indicies = np.array(action[:3])
            rot_and_grip_indicies = np.array(action[3:6])
            is_gripper_open = action[6]

            bounds = SCENE_BOUNDS
            res = (bounds[3:] - bounds[:3]) / 100
            attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
            quat = discrete_euler_to_quaternion(rot_and_grip_indicies)
            
            continuous_action = np.concatenate([
                attention_coordinate,
                quat,
                [is_gripper_open],
                [1],
            ])
            output.append(continuous_action)
        
        # get subsequent predicted actions
        return output[:26]
        

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        # inference
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.actions) == 0:
            output = self._postprocess(output_text)
            self.actions = output
            
        continuous_action = self.actions.pop(0)

        self.step += 1
        
        # copy_obs = {k: v.cpu() for k, v in observation.items()}
        copy_obs={}
        for k, v in observation.items():
            # print(k, type(v))
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

    def load_weights(self, savedir: str, components={}):
        # no weight to load
        # only build task handler
        self.savedir = savedir
        
        self.components=components

        from form_icl_demonstrations_crosstask_ranking import create_task_handler

        self.handler = create_task_handler(self.task_name)
        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return