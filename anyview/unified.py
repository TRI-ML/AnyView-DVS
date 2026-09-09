# minimal reader for the unified scene layout (frames or videos variant).
'''
One directory per scene:

  metadata.json                info.name, info.storage ('frames' | 'videos'), cameras [...],
                               resolution [H, W] (or null when cameras differ), num_frames,
                               framerate, rgb.extension, extrinsics.transform ('cam2world' |
                               'world2cam'), optional specific {...} (dataset-specific fields)
  rgb/<cam>/0000000000.jpg     frames variant, one image per frame (jpg or png)
  rgb/<cam>.mp4                videos variant, all-intra h264; video frame i == lowdim row i
  lowdim/<cam>/0000000000.npz  frames variant: camera, timestep, intrinsics (3, 3), extrinsics (4, 4)
  lowdim/<cam>.npz             videos variant, stacked over frames: camera (T,), timestep (T,),
                               intrinsics (T, 3, 3), extrinsics (T, 4, 4)

Extrinsics are returned as cam2world regardless of the on-disk convention. Frame indexing is
positional (row i of the lowdim arrays belongs to image / video frame i); the framerate field
is informational only.
'''

import json
import os

import numpy as np

FRAME_NAME = '{:010d}'
IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png')


def read_metadata(scene_dir):
    path = os.path.join(scene_dir, 'metadata.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Not a unified scene (no metadata.json): {scene_dir}')
    with open(path, 'r') as f:
        metadata = json.load(f)
    return metadata


class UnifiedScene:
    '''
    Lazy accessor for one unified scene (frames or videos storage).
    '''

    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.metadata = read_metadata(scene_dir)
        self.cameras = list(self.metadata['cameras'])
        self.num_frames = int(self.metadata['num_frames'])
        self.rgb_extension = str(self.metadata.get('rgb', {}).get('extension', 'jpg')).lower()
        storage = self.metadata.get('info', {}).get('storage', None)
        if storage is None:
            storage = 'videos' if self.rgb_extension == 'mp4' else 'frames'
        if storage not in ('frames', 'videos'):
            raise ValueError(f'{scene_dir}: unknown storage {storage!r}')
        self.storage = storage
        transform = self.metadata.get('extrinsics', {}).get('transform', 'cam2world')
        if transform not in ('cam2world', 'world2cam'):
            raise ValueError(f'{scene_dir}: unknown extrinsics.transform {transform!r}')
        self.extrinsics_transform = transform
        self.specific = self.metadata.get('specific', {}) or {}

    # ---- rgb ----

    def rgb_path(self, cam):
        if self.storage == 'videos':
            path = os.path.join(self.scene_dir, 'rgb', f'{cam}.mp4')
        else:
            path = os.path.join(self.scene_dir, 'rgb', cam)
        return path

    def has_rgb(self, cam):
        return os.path.exists(self.rgb_path(cam))

    def load_rgb(self, cam, frame_indices=None):
        '''
        :param frame_indices (list of int or None): None = all frames of the scene.
        :return video: (T, H, W, 3) uint8 numpy array.
        '''
        import imageio.v2 as imageio

        if frame_indices is None:
            frame_indices = list(range(self.num_frames))
        path = self.rgb_path(cam)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{self.scene_dir}: no rgb for camera {cam} ({path})')

        if self.storage == 'videos':
            reader = imageio.get_reader(path)
            decoded = []
            last = max(frame_indices)
            for (i, frame) in enumerate(reader):
                decoded.append(np.asarray(frame)[..., 0:3])
                if i >= last:
                    break
            reader.close()
            if len(decoded) <= last:
                raise ValueError(f'{path} has {len(decoded)} frames, need index {last}')
            frames = [decoded[i] for i in frame_indices]
        else:
            frames = []
            for i in frame_indices:
                fp = os.path.join(path, FRAME_NAME.format(i) + '.' + self.rgb_extension)
                if not os.path.isfile(fp):
                    raise FileNotFoundError(f'Missing frame {fp}')
                frames.append(np.asarray(imageio.imread(fp))[..., 0:3])
        video = np.stack(frames, axis=0)
        return video

    # ---- lowdim ----

    def load_lowdim(self, cam, frame_indices=None):
        '''
        :return (intrinsics, cam2world, timesteps): (T, 3, 3) float32, (T, 4, 4) float32, (T,) int64.
        '''
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        if self.storage == 'videos':
            fp = os.path.join(self.scene_dir, 'lowdim', f'{cam}.npz')
            if not os.path.isfile(fp):
                raise FileNotFoundError(f'{self.scene_dir}: no lowdim for camera {cam} ({fp})')
            data = np.load(fp)
            intr_all = np.asarray(data['intrinsics'])
            extr_all = np.asarray(data['extrinsics'])
            ts_all = np.asarray(data['timestep'])
            intrinsics = intr_all[frame_indices]
            extrinsics = extr_all[frame_indices]
            timesteps = ts_all[frame_indices]
        else:
            intr_list = []
            extr_list = []
            ts_list = []
            for i in frame_indices:
                fp = os.path.join(self.scene_dir, 'lowdim', cam, FRAME_NAME.format(i) + '.npz')
                if not os.path.isfile(fp):
                    raise FileNotFoundError(f'Missing lowdim {fp}')
                data = np.load(fp)
                if 'intrinsics' not in data or 'extrinsics' not in data:
                    raise KeyError(f'{fp} has no intrinsics/extrinsics (keys: {list(data.keys())})')
                intr_list.append(np.asarray(data['intrinsics']))
                extr_list.append(np.asarray(data['extrinsics']))
                ts_list.append(int(data['timestep']))
            intrinsics = np.stack(intr_list, axis=0)
            extrinsics = np.stack(extr_list, axis=0)
            timesteps = np.asarray(ts_list)

        intrinsics = intrinsics.astype(np.float32)
        extrinsics = extrinsics.astype(np.float64)
        assert intrinsics.shape[1:] == (3, 3), intrinsics.shape
        assert extrinsics.shape[1:] == (4, 4), extrinsics.shape
        if self.extrinsics_transform == 'world2cam':
            extrinsics = np.linalg.inv(extrinsics)
        cam2world = extrinsics.astype(np.float32)
        timesteps = timesteps.astype(np.int64)
        return (intrinsics, cam2world, timesteps)


def world2cam_from_cam2world(cam2world):
    '''
    (T, 4, 4) float32 -> (T, 4, 4) float32, inverted in float64.
    '''
    world2cam = np.linalg.inv(np.asarray(cam2world, dtype=np.float64)).astype(np.float32)
    return world2cam
