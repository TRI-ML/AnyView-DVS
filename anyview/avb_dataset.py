# torch Dataset for AnyViewBench episodes (unified scene layout (frames)).

import json
import os

import torch
from torch.utils.data import Dataset

from anyview.unified import UnifiedScene, world2cam_from_cam2world


class AVBDataset(Dataset):
    '''
    Loader for the AnyView Benchmark (AVB):

    root/
      splits.json                   {"splits": {split: {"episodes": [...], "redistributable":
                                    bool, "dir": group}}, "groups": {group: [splits]}}
      name_mapping.json             episode -> original dataset / sequence identifiers
      <group>/<episode>/            one unified scene per episode (group = paper table row;
                                    the direction, e.g. FL / LR, is part of the episode name)
        metadata.json               unified metadata; specific.roles = {input: cam1, target: cam0},
                                    specific.scale_factor, specific.anyview_zero_shot, specific.source
        rgb/cam1/0000000000.jpg ... input-view frames (conditioning video, native res)
        rgb/cam0/0000000000.jpg ... target-view GT frames (benchmark only)
        lowdim/cam{0,1}/*.npz       per-frame intrinsics (3x3) + cam2world (4x4)

    Each sample yields raw uint8 frames (T, H, W, 3) per view plus intrinsics and per-frame
    world2cam matrices; no resizing happens here (the eval script owns preprocessing). Frame
    count T is uniform within an episode but varies across splits.
    '''

    def __init__(self, root, split=None, episodes=None, require_target=True):
        '''
        :param root (str): path to the AVB root directory (contains splits.json).
        :param split (str or list or None): restrict to one split or a list of splits.
        :param episodes (list or None): explicit episode names (overrides split filtering).
        :param require_target (bool): if True, raise when the target rgb is missing.
        '''
        self.root = root
        self.require_target = require_target

        splits_fp = os.path.join(root, 'splits.json')
        if not os.path.isfile(splits_fp):
            raise FileNotFoundError(f'Missing splits.json at {splits_fp}')
        with open(splits_fp, 'r') as f:
            splits_info = json.load(f)
        raw_splits = splits_info['splits']
        # split -> {'episodes': [...], 'redistributable': bool}; accept the bare-list form too
        self.splits = {}
        for name, val in raw_splits.items():
            if isinstance(val, list):
                self.splits[name] = {'episodes': val, 'redistributable': True}
            else:
                self.splits[name] = val
        self.split_episodes = {k: v['episodes'] for (k, v) in self.splits.items()}
        # Episode directories resolve via the index ('dir' = group folder); a flat
        # <split>/ layout remains the fallback for indexes without 'dir'.
        self._split_dir = {k: v.get('dir', k) for (k, v) in self.splits.items()}
        self.groups = splits_info.get('groups', {})  # group name -> [split names]
        # A benchmark tree may include only some groups; splits whose folder is absent are
        # dropped here with a notice, so nothing fails later.
        for (name, folder) in list(self._split_dir.items()):
            if not os.path.isdir(os.path.join(root, folder)):
                print(f'[AVBDataset] split {name} not present in {root} (no {folder}/); skipping')
                del self.splits[name]
                del self.split_episodes[name]
                del self._split_dir[name]
        self.groups = {g: [s_ for s_ in m if s_ in self.splits] for (g, m) in self.groups.items()}
        self.groups = {g: m for (g, m) in self.groups.items() if m}

        name_map_fp = os.path.join(root, 'name_mapping.json')
        self.name_mapping = {}
        if os.path.isfile(name_map_fp):
            with open(name_map_fp, 'r') as f:
                self.name_mapping = json.load(f)

        if isinstance(split, str):
            split = [split]
        self._episode_split = {}
        for split_name, ep_names in self.split_episodes.items():
            for ep in ep_names:
                self._episode_split[ep] = split_name

        if episodes is not None:
            selected = list(episodes)
        else:
            selected = []
            for split_name, ep_names in self.split_episodes.items():
                if split is not None and split_name not in split:
                    continue
                selected.extend(ep_names)
        if len(selected) == 0:
            raise ValueError(f'No episodes selected (root={root}, split={split})')
        for ep in selected:
            if ep not in self._episode_split:
                raise ValueError(f'Unknown episode {ep} (not listed in splits.json)')
        self.episodes = selected

    def __len__(self):
        return len(self.episodes)

    def episode_dir(self, episode):
        '''
        Absolute directory of one episode (resolved via the splits.json index).
        '''
        split = self._episode_split[episode]
        path = os.path.join(self.root, self._split_dir[split], episode)
        return path

    def has_pixels(self, episode):
        '''
        False for episodes of restricted splits that come without rgb/ (public tier).
        '''
        scene = UnifiedScene(self.episode_dir(episode))
        roles = scene.specific.get('roles', {'input': 'cam1', 'target': 'cam0'})
        present = scene.has_rgb(roles['input'])
        return present

    def __getitem__(self, index):
        episode = self.episodes[index]
        split = self._episode_split[episode]
        ep_dir = self.episode_dir(episode)

        scene = UnifiedScene(ep_dir)
        roles = scene.specific.get('roles', {'input': 'cam1', 'target': 'cam0'})
        cam_in = roles['input']
        cam_out = roles['target']

        if not scene.has_rgb(cam_in):
            raise FileNotFoundError(
                f'Episode {episode} has no rgb/{cam_in} frames. This split comes without pixels '
                f'(source dataset license does not permit redistribution). Rebuild it from '
                f'an official download with scripts/prepare_restricted.py; the episode\'s '
                f'fetch_manifest.json documents the exact sequence, cameras, and frame indices.')
        input_video = torch.from_numpy(scene.load_rgb(cam_in))  # (T, H, W, 3) uint8

        target_video = None
        if scene.has_rgb(cam_out):
            target_video = torch.from_numpy(scene.load_rgb(cam_out))
        elif self.require_target:
            raise FileNotFoundError(f'Missing rgb/{cam_out} in {ep_dir} (require_target=True)')

        (K_in, c2w_in, _) = scene.load_lowdim(cam_in)
        (K_out, c2w_out, _) = scene.load_lowdim(cam_out)
        w2c_in = torch.from_numpy(world2cam_from_cam2world(c2w_in))
        w2c_out = torch.from_numpy(world2cam_from_cam2world(c2w_out))

        T_in = input_video.shape[0]
        if w2c_in.shape[0] != T_in:
            raise ValueError(f'{episode}: input has {T_in} frames but {w2c_in.shape[0]} poses')
        if target_video is not None and w2c_out.shape[0] != target_video.shape[0]:
            raise ValueError(f'{episode}: target has {target_video.shape[0]} frames but '
                             f'{w2c_out.shape[0]} poses')

        # Intrinsics are constant within an episode (frame 0 is representative).
        sample = {
            'episode': episode,
            'split': split,
            'input': input_video,                                  # (T, H1, W1, 3) uint8
            'target_gt': target_video,                             # (T, H0, W0, 3) uint8 or None
            'intrinsics_input': torch.from_numpy(K_in[0]),         # (3, 3) float32
            'intrinsics_target': torch.from_numpy(K_out[0]),       # (3, 3) float32
            'world2cam_input': w2c_in,                             # (T, 4, 4) float32
            'world2cam_target': w2c_out,                           # (T, 4, 4) float32
            'scale_factor': scene.specific.get('scale_factor', None),
            'source': scene.specific.get('source', {}),
        }
        return sample
