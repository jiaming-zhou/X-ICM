import copy
import glob
import itertools
import os
import pickle
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from PIL import Image
from pyrep.objects import VisionSensor
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils import _image_to_float_array, normalize_quaternion, point_to_voxel_index, quaternion_to_discrete_euler, CAMERAS
from utils import encode_img
import json
from main import PROJECT_ROOT

from rlbench_inference_dynamics_diffusion import *

seen_sim_name_to_real_name={}

demo_num_per_icl=1
demo_nums = 200


seen_path=os.path.join(PROJECT_ROOT, "data/seen_tasks/train")
unseen_path = os.path.join(PROJECT_ROOT, "data/unseen_tasks/test") 
f=open(os.path.join(PROJECT_ROOT, "data/dynamics_diffusion/all_diffusion_features.pkl"), 'rb')
all_diffusion_features=pickle.load(f)
f.close()

class base_task_handler:
    def __init__(self, sim_name_to_real_name):
            self.sim_name_to_real_name = sim_name_to_real_name
            self.save_root = os.path.join(unseen_path, type(self).__name__)
            self.num_demos = demo_num_per_icl
            print(f"Task handler {type(self).__name__} using demonstrations from {self.save_root}")
            random.seed(42)

    def get_user_prompt_ranking(self, mask_dict, mask_id_to_sim_name, point_cloud_dict, custom_num_demos=-1, taskname='None', image_path=None, seed=0, ranking_metric="lang_vis.out"):
        assert os.path.exists(self.save_root), f"Cannot find save root {self.save_root}"
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos
        
        mask_id_to_real_name = {mask_id: self.sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                            if name in self.sim_name_to_real_name}
        obs = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict, taskname=taskname, cross_task_eval=1)

        all_demo_paths = all_diffusion_features['all_demo_paths']
        
        if "lang_vis.out" in ranking_metric:
            
            all_input_image_feats = all_diffusion_features['all_input_image_feats']
            all_output_image_feats = all_diffusion_features['all_output_image_feats']
            all_prompt_feats = all_diffusion_features['all_prompt_feats']

            query_input_image_feat, query_output_image_feat, \
                query_prompt_feat = extract_diffusion_features(image_path, taskname)

            query_feat = np.concatenate([query_prompt_feat, query_output_image_feat])  # (2048,)
            memory_feat = np.concatenate([all_prompt_feats, all_output_image_feats], axis=1)  # (M, 2048)

            similarity = np.dot(memory_feat, query_feat)

            top_indices = np.argsort(similarity)[::-1]

            selected_indices = top_indices[:self.num_demos]
        
        elif "random" in ranking_metric:
            selected_indices = random.sample(range(len(all_demo_paths)), self.num_demos)
        else:
            raise ValueError(f"Invalid ranking metric: {ranking_metric}")


        output = ""
        for i, selected_idx in enumerate(selected_indices):
            icl_episode_path=all_demo_paths[selected_idx]
            icl_task_name=icl_episode_path.split('/')[-4]
            icl_episode_id=int(icl_episode_path.split('/')[-1][7:])
            print("the %d-th icl: "%(i+1), ranking_metric, icl_episode_path)
            train_demos = get_stored_demos_crosstask(seen_path, icl_task_name, icl_episode_id, 1, seen_sim_name_to_real_name[icl_task_name])

            for epi in train_demos:
                output += f"{epi[0]}>{epi[1]}, "
        
        return output + obs + ">"



    def save_in_context_demonstrations(self, custom_num_demos=-1):
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos

        train_demos = get_stored_demos(unseen_path, type(self).__name__, demo_nums, self.sim_name_to_real_name)

        # iterate over demo_nums demonstrations, each time take self.num_demos demonstrations
        for i, start_idx in enumerate(range(0, len(train_demos) - self.num_demos + 1, self.num_demos)):
            if start_idx + self.num_demos <= len(train_demos):
                output = ""
                for epi in train_demos[start_idx:start_idx+self.num_demos]:
                    output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(unseen_path, type(self).__name__, f"demonstrations_{self.num_demos}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'{i}.txt'), "w") as f:
                    f.write(output)
    
    def save_in_context_demonstrations_crosstask(self, custom_num_demos=-1):
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos
        
        train_tasknames=os.listdir(seen_path)
        
        # iterate over demo_nums demonstrations, each time take self.num_demos demonstrations
        for i, start_idx in enumerate(range(0, demo_nums - self.num_demos + 1, self.num_demos)):
            if start_idx + self.num_demos <= demo_nums:
                output = ""

                for taskname in train_tasknames:
                    train_demos = get_stored_demos_crosstask(seen_path, taskname, start_idx, self.num_demos, seen_sim_name_to_real_name[taskname])

                    for epi in train_demos:
                        output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(unseen_path, type(self).__name__, f"demonstrations.crosstask_{self.num_demos}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'{i}.txt'), "w") as f:
                    f.write(output)

    


class close_jar(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "jar_lid0": "lid",
            "jar0": "jar",
        }
        super().__init__(sim_name_to_real_name)


class open_drawer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "drawer_bottom": "drawer",
        }
        super().__init__(sim_name_to_real_name)

class slide_block_to_color_target(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target1": "target",
            "block": "block"
        }

        super().__init__(sim_name_to_real_name)

class sweep_to_dustpan_of_size(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "dustpan_tall": "dustpan",
            "broom_holder": "broom holder"
        }
        super().__init__(sim_name_to_real_name)

class meat_off_grill(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "chicken_visual": "chicken",
            "grill_visual": "grill"
        }
        super().__init__(sim_name_to_real_name)

class turn_tap(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "tap_left_visual": "left tap",
            "tap_right_visual": "right tap"
        }
        super().__init__(sim_name_to_real_name)

class put_item_in_drawer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "item": "item",
            "drawer_frame": "drawer"
        }
        super().__init__(sim_name_to_real_name)

class stack_blocks(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "stack_blocks_target0": "first block",
            "stack_blocks_target1": "second block",
            "stack_blocks_target2": "third block",
            "stack_blocks_target3": "fourth block",
            "stack_blocks_target_plane": "plane",
        }
        super().__init__(sim_name_to_real_name)

class light_bulb_in(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "bulb0": "blub",
            "lamp_screw": "lamp screw",
        }
        super().__init__(sim_name_to_real_name)

class put_money_in_safe(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "dollar_stack": "money",
            "safe_body": "shelf",
        }
        super().__init__(sim_name_to_real_name)

class place_wine_at_rack_location(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "wine_bottle_visual": "wine",
            "rack_top_visual": "rack",
        }
        # super().__init__(system_prompt, sim_name_to_real_name, num_demos, num_keypoints)
        super().__init__(sim_name_to_real_name)


class put_groceries_in_cupboard(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cupboard": "cupboard",
            "crackers_visual": "cracker",
        }
        super().__init__(sim_name_to_real_name)


class place_shape_in_shape_sorter(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cube": "cube",
            # "shape_sorter": "shape sorter",
            "shape_sorter_visual": "shape sorter",
        }
        super().__init__(sim_name_to_real_name)

class push_buttons(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target_button_wrap0": "button",
        }
        super().__init__(sim_name_to_real_name)

class insert_onto_square_peg(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "square_ring": "ring",
            "pillar0": "first spok",
            "pillar1": "second spok",
            "pillar2": "third spok"
            }
        super().__init__(sim_name_to_real_name)

class stack_cups(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cup1_visual": "first cup",
            "cup2_visual": "second cup",
            "cup3_visual": "third cup",
        }
        super().__init__(sim_name_to_real_name)

class place_cups(base_task_handler):
    def __init__(self):

        sim_name_to_real_name = {
            "mug_visual1": "first cup",
            "mug_visual0": "second cup",
            "mug_visual2": "third cup",
            "mug_visual3": "forth cup",
            "place_cups_holder_spoke0": "first holder",
            "place_cups_holder_spoke1": "second holder",
            "place_cups_holder_spoke2": "third holder"
        }
        super().__init__(sim_name_to_real_name)



class reach_and_drag(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "stick": "stick",
            "cube": "cube"
        }
        super().__init__(sim_name_to_real_name)



########## zero shot tasks ################
class basketball_in_hoop(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "basket_ball_hoop_visual": "hoop",
            "ball": "ball"
        }
        super().__init__(sim_name_to_real_name)

class scoop_with_spatula(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "Cuboid": "cube",
            "spatula_visual": "spatula"
        }
        super().__init__(sim_name_to_real_name)

class straighten_rope(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "head": "rope head",
            "head_target": "rope head target",
            "head_tail": "rope head tail",
            "tail": "rope tail",
        }
        super().__init__(sim_name_to_real_name)

class turn_oven_on(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "oven_door": "oven door",
            "oven_knob_8": "first oven knob",
            "oven_knob_9": "second oven knob",
        }
        super().__init__(sim_name_to_real_name)

class beat_the_buzz(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "Cuboid": "cube",
            "wand_visual": "pole",
            "wand_visual_sub": "pole head"
        }
        super().__init__(sim_name_to_real_name)

class water_plants(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "waterer_visual": "waterer",
            "plant_visual": "plant",
            "base_visual": "waterer base"
        }
        super().__init__(sim_name_to_real_name)

class unplug_charger(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "charger_visual": "charger",
            "task_wall": "wall",
        }
        super().__init__(sim_name_to_real_name)

class phone_on_base(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "phone_visual": "phone",
            "phone_case_visual": "phone case",
        }
        super().__init__(sim_name_to_real_name)

class toilet_seat_down(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "toilet_seat_up_toilet": "toilet seat up_toilet",
            "toilet_seat_up_seat": "toilet seat up_seat",
            "toilet": "toilet",
        }
        super().__init__(sim_name_to_real_name)

class lamp_off(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "push_button_target": "button",
            "target_button_topPlate": "button topPlate",
            "target_button_wrap": "button wrap",
        }
        super().__init__(sim_name_to_real_name)

class lamp_on(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "push_button_target": "button",
            "target_button_topPlate": "button topPlate",
            "target_button_wrap": "button wrap",
        }
        super().__init__(sim_name_to_real_name)

class put_books_on_bookshelf(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "book0_visual": "first book",
            "book1_visual": "second book",
            "book2_visual": "third book",
            "bookshelf_visual": "bookshelf",
        }
        super().__init__(sim_name_to_real_name)

class put_umbrella_in_umbrella_stand(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "umbrella_visual": "umbrella",
            "stand_visual": "umbrella stand",
        }
        super().__init__(sim_name_to_real_name)

class open_grill(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "grill_visual": "grill",
            "lid_visual": "lid",
            "handle_visual": "handle",
        }
        super().__init__(sim_name_to_real_name)

class put_rubbish_in_bin(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "rubbish_visual": "rubbish",
            "bin_visual": "bin",
        }
        super().__init__(sim_name_to_real_name)

class take_usb_out_of_computer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "computer_visual": "computer",
            "usb_visual": "usb",
        }
        super().__init__(sim_name_to_real_name)

class take_lid_off_saucepan(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "saucepan_visual": "saucepan",
            "saucepan_lid_visual": "saucepan lid",
        }
        super().__init__(sim_name_to_real_name)

class take_plate_off_colored_dish_rack(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "plate_visual": "plate",
            "dish_rack_pillar0": "first dish rack",
            "dish_rack_pillar1": "second dish rack",
        }
        super().__init__(sim_name_to_real_name)

class close_fridge(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "fridge_base_visual": "fridge base",
            "door_top_visual": "fridge top door",
            "door_bottom_visual": "fridge bottom door",
        }
        super().__init__(sim_name_to_real_name)

class close_microwave(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "microwave_door": "microwave door",
            "microwave_frame_vis": "microwave frame",
        }
        super().__init__(sim_name_to_real_name)

class close_laptop_lid(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "lid_visual": "lid",
            "laptop_holder": "laptop holder",
            "base_visual":"laptop base"
        }
        super().__init__(sim_name_to_real_name)

class put_toilet_roll_on_stand(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "toilet_roll_visual": "toilet roll",
            "holder_visual": "holder",
            "stand_base": "stand_base",
        }
        super().__init__(sim_name_to_real_name)

class put_knife_on_chopping_board(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "knife_visual": "knife",
            "chopping_board_visual": "chopping board",
        }
        super().__init__(sim_name_to_real_name)


seen_task_name_to_handler = {
    "close_jar": close_jar,
    "open_drawer": open_drawer,
    "slide_block_to_color_target": slide_block_to_color_target,
    "sweep_to_dustpan_of_size": sweep_to_dustpan_of_size,
    "meat_off_grill": meat_off_grill,
    "turn_tap": turn_tap,
    "put_item_in_drawer": put_item_in_drawer,
    "stack_blocks": stack_blocks,
    "light_bulb_in": light_bulb_in,
    "put_money_in_safe": put_money_in_safe,
    "place_wine_at_rack_location": place_wine_at_rack_location, 
    "put_groceries_in_cupboard": put_groceries_in_cupboard,
    "place_shape_in_shape_sorter": place_shape_in_shape_sorter,
    "push_buttons": push_buttons,
    "stack_cups": stack_cups,
    "place_cups": place_cups,
    "insert_onto_square_peg":insert_onto_square_peg,
    "reach_and_drag": reach_and_drag,
    }

unseen_task_name_to_handler = {
    "put_toilet_roll_on_stand": put_toilet_roll_on_stand,
    "put_knife_on_chopping_board": put_knife_on_chopping_board,
    "close_fridge": close_fridge,
    "close_microwave": close_microwave,
    "close_laptop_lid":close_laptop_lid,
    "phone_on_base": phone_on_base,
    "toilet_seat_down": toilet_seat_down,
    "lamp_off": lamp_off,
    "lamp_on": lamp_on,
    "put_books_on_bookshelf": put_books_on_bookshelf,
    "put_umbrella_in_umbrella_stand": put_umbrella_in_umbrella_stand,
    "open_grill": open_grill,
    "put_rubbish_in_bin": put_rubbish_in_bin,
    "take_usb_out_of_computer": take_usb_out_of_computer,
    "take_lid_off_saucepan": take_lid_off_saucepan,
    "take_plate_off_colored_dish_rack": take_plate_off_colored_dish_rack,
    "basketball_in_hoop":basketball_in_hoop,
    "scoop_with_spatula":scoop_with_spatula,
    "straighten_rope":straighten_rope,
    "turn_oven_on":turn_oven_on,
    "beat_the_buzz": beat_the_buzz,
    "water_plants": water_plants,
    "unplug_charger": unplug_charger
    }


task_name_to_handler = unseen_task_name_to_handler
def create_task_handler(task_name):
    return task_name_to_handler[task_name]()


train_tasknames=os.listdir(seen_path)
for taskname in train_tasknames:
    handler = seen_task_name_to_handler[taskname]()
    seen_sim_name_to_real_name[taskname]=handler.sim_name_to_real_name
    del handler


# discretize translation, rotation, gripper open
def _get_action(
        obs_tp1,
        obs_tm1):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat)
    trans_indicies = []
    ignore_collisions = int(obs_tm1.ignore_collisions)

    index = point_to_voxel_index(
        obs_tp1.gripper_pose[:3])
    trans_indicies.extend(index.tolist())

    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies + rot_and_grip_indicies

def _get_point_cloud_dict(epis_path, idx):
    # This function gets the point cloud using the same operations as PerAct Colab Tutorial
    DEPTH_SCALE = 2**24 - 1
    point_cloud_dict = {}
    for camera_type in CAMERAS:
        with open(os.path.join(epis_path, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        cam_extrinsics = demo[idx].misc[f"{camera_type}_camera_extrinsics"]
        cam_intrinsics = demo[idx].misc[f"{camera_type}_camera_intrinsics"]
        cam_depth = _image_to_float_array(Image.open(os.path.join(epis_path, f"{camera_type}_depth", f"{idx}.png")), DEPTH_SCALE)
        near = demo[idx].misc[f"{camera_type}_camera_near"]
        far = demo[idx].misc[f"{camera_type}_camera_far"]
        cam_depth = (far - near) * cam_depth + near
        point_cloud_dict[camera_type] = VisionSensor.pointcloud_from_depth_and_camera_params(cam_depth, cam_extrinsics, cam_intrinsics) # reconstructed 3D point cloud in world coordinate frame

    return point_cloud_dict

def _get_mask_dict(epis_path, idx):
    mask_dict = {}
    for camera in CAMERAS:
        img = Image.open(os.path.join(epis_path, f"{camera}_mask", f"{idx}.png"))
        rgb_mask = np.array(img, dtype=int)
        mask_dict[camera] = rgb_mask[:, :, 0] + rgb_mask[:, :, 1]*256 + rgb_mask[:, :, 2]*256*256
    return mask_dict

def _get_mask_id_to_name_dict(epis_path, idx):
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        low_dim_obs = pickle.load(f)
    mask_id_to_name_dict = {}
    for camera in CAMERAS:
        mask_id_to_name_dict[camera] = low_dim_obs[idx].misc[f"{camera}_mask_id_to_name"]
    return mask_id_to_name_dict

# add individual data points to replay
def _add_keypoints_to_replay(
        buffer,
        i,
        demo,
        episode_keypoints,
        epis_path_depth,
        epis_path_char,
        sim_name_to_real_name,
        cross_task_eval
    ):
    prev_action = None
    cur_index = i

    mask_dict = _get_mask_dict(epis_path_char, cur_index)

    mask_id_to_sim_name_dict = _get_mask_id_to_name_dict(epis_path_char, cur_index)
    point_cloud_dict = _get_point_cloud_dict(epis_path_depth, cur_index)
    
    mask_id_to_sim_name = {}
    for camera in CAMERAS:
        mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera])

    mask_id_to_real_name = {mask_id: sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                        if name in sim_name_to_real_name}

    avg_coord = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict, demo.language_descriptions[0], cross_task_eval)

    buffer.append(avg_coord)
    actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(
            obs_tp1, obs_tp1)

        actions.append(action)
    
    buffer.append(actions)

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo, delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    #print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def get_stored_demos(dataset_root, task_name, amount, sim_name_to_real_name):
    total_num_keypoints = 0
    buffer = []
    task_root = os.path.join(dataset_root, task_name, 'all_variations', 'episodes')

    for epi_id in tqdm(range(amount)):
        epis_path_depth = os.path.join(task_root, f'episode{epi_id}')
        epis_path_char = os.path.join(task_root, f'episode{epi_id}')

        with open(os.path.join(epis_path_depth, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path_depth, 'variation_number.pkl'), 'rb') as f:
            demo.variation_number = pickle.load(f)

        # language description
        with open(os.path.join(epis_path_depth, 'variation_descriptions.pkl'), 'rb') as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = _keypoint_discovery(demo)

        tmp = []
        _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path_depth, epis_path_char, sim_name_to_real_name, cross_task_eval=0)
        tmp.append(f"{epis_path_depth}/front_rgb/0.png")
        buffer.append(tmp)

    print("Average number of steps: ", sum([len(each[1]) for each in buffer])/len(buffer))
    return buffer


def get_stored_demos_crosstask(dataset_root, task_name, start_idx, num_demos, sim_name_to_real_name, cross_task_eval=1):
    total_num_keypoints = 0
    buffer = []
    task_root = os.path.join(dataset_root, task_name, 'all_variations', 'episodes')

    for epi_id in tqdm(range(start_idx, start_idx+num_demos)):
        epis_path_depth = os.path.join(task_root, f'episode{epi_id}')
        epis_path_char = os.path.join(task_root, f'episode{epi_id}')

        with open(os.path.join(epis_path_depth, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path_depth, 'variation_number.pkl'), 'rb') as f:
            demo.variation_number = pickle.load(f)

        # language description
        with open(os.path.join(epis_path_depth, 'variation_descriptions.pkl'), 'rb') as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = _keypoint_discovery(demo)

        tmp = []
        _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path_depth, epis_path_char, sim_name_to_real_name, cross_task_eval)
        tmp.append(f"{epis_path_depth}/front_rgb/0.png")
        buffer.append(tmp)
    # print("Average number of steps: ", sum([len(each[1]) for each in buffer])/len(buffer))
    return buffer



def form_obs(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict,
    taskname='None',
    cross_task_eval=0):
    
    # convert object id to char and average and discretize point cloud per object
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}

    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name:
            continue
        avg_point_list = []
        for camera in CAMERAS:
            mask = mask_dict[camera]
            point_cloud = point_cloud_dict[camera]
            if not np.any(mask == mask_id):
                continue
            avg_point_list.append(np.mean(point_cloud[mask == mask_id].reshape(-1, 3), axis = 0))

        avg_point = sum(avg_point_list) / len(avg_point_list)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    
    if cross_task_eval:
        return "['instruction': "+taskname+", "+str(real_name_to_avg_coord)+"]"
    return str(real_name_to_avg_coord)




if __name__ == "__main__":
    pass

