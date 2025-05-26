from scipy.spatial.transform import Rotation
import numpy as np


SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
ROTATION_RESOLUTION = 5
VOXEL_SIZE = 100
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
IMAGE_SIZE = [128, 128]

# From https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
def _image_to_float_array(image, scale_factor):
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array

def quaternion_to_discrete_euler(quaternion):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / ROTATION_RESOLUTION)).astype(int)
    disc[disc == int(360 / ROTATION_RESOLUTION)] = 0
    return disc

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)

def point_to_voxel_index(
        point: np.ndarray):
    bb_mins = np.array(SCENE_BOUNDS[0:3])[None]
    bb_maxs = np.array(SCENE_BOUNDS[3:])[None]
    dims_m_one = np.array([VOXEL_SIZE] * 3)[None] - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([VOXEL_SIZE] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy.reshape(point.shape)

def discrete_euler_to_quaternion(discrete_euler):
    euluer = (discrete_euler * ROTATION_RESOLUTION) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()


def convert_to_euler(rpy):
    euluer = Rotation.from_quat(rpy).as_euler('xyz', degrees=False)
    return euluer

def euler_to_quaternion(euler, degree=False):
    if degree:
        return Rotation.from_euler("xyz", euler, degrees=True).as_quat()
    else:
        return Rotation.from_euler("xyz", euler).as_quat()


import base64
def encode_img(image_path):
    """Encode image to Base64 format."""
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read())
    return f"data:image;base64,{encoded_image.decode('utf-8')}"


    