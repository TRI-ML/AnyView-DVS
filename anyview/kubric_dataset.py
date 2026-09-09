# training dataset over unified Kubric-5D scenes (videos or frames variant).

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from anyview.unified import UnifiedScene, read_metadata, world2cam_from_cam2world

KUBRIC_SCALE_FACTOR = 1.0 / 32.0  # synthetic scenes are oversized (matches the checkpoint)


def list_scenes(root):
    '''
    Scene directories directly under root that carry a metadata.json (sorted).
    '''
    scenes = []
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if os.path.isfile(os.path.join(scene_dir, 'metadata.json')):
            scenes.append(scene_dir)
    return scenes


class KubricDVSDataset(Dataset):
    '''
    One sample = one random camera pair from one scene, over a random window of num_frames
    consecutive frames: view 1 (input) and view 0 (target) as raw uint8 frames plus intrinsics
    and world2cam per view, in the same key layout as AVBDataset samples. Resizing and pose
    re-anchoring happen in the training script.
    '''

    def __init__(self, root, num_frames=41, scenes=None, seed=None):
        self.root = root
        self.num_frames = int(num_frames)
        self.scene_dirs = scenes if scenes is not None else list_scenes(root)
        if len(self.scene_dirs) == 0:
            raise ValueError(f'No unified scenes under {root}')
        self.rng = random.Random(seed)
        meta = read_metadata(self.scene_dirs[0])
        self.scale_factor = KUBRIC_SCALE_FACTOR
        if int(meta['num_frames']) < self.num_frames:
            raise ValueError(f'Scenes have {meta["num_frames"]} frames, need {self.num_frames}')

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, index):
        scene = UnifiedScene(self.scene_dirs[index])
        (cam_in, cam_out) = self.rng.sample(scene.cameras, 2)
        start = self.rng.randint(0, scene.num_frames - self.num_frames)
        indices = list(range(start, start + self.num_frames))

        input_video = torch.from_numpy(scene.load_rgb(cam_in, indices))
        target_video = torch.from_numpy(scene.load_rgb(cam_out, indices))
        (K_in, c2w_in, _) = scene.load_lowdim(cam_in, indices)
        (K_out, c2w_out, _) = scene.load_lowdim(cam_out, indices)

        sample = {
            'episode': f'{os.path.basename(self.scene_dirs[index])}_{cam_in}_to_{cam_out}_f{start}',
            'input': input_video,                                  # (T, H, W, 3) uint8
            'target_gt': target_video,                             # (T, H, W, 3) uint8
            'intrinsics_input': torch.from_numpy(K_in[0]),
            'intrinsics_target': torch.from_numpy(K_out[0]),
            'world2cam_input': torch.from_numpy(world2cam_from_cam2world(c2w_in)),
            'world2cam_target': torch.from_numpy(world2cam_from_cam2world(c2w_out)),
            'scale_factor': self.scale_factor,
        }
        return sample
