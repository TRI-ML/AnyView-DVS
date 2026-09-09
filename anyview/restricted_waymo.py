# rebuild AnyViewBench Waymo episode pixels from the official Waymo Open Dataset download.
'''
The Waymo splits of AnyViewBench (waymo_FL / waymo_FR / waymo_LF / waymo_RF) come without rgb/.
rebuild_episode() rebuilds rgb/cam0 (target view) and rgb/cam1 (input view) of one episode
from the user's own official download of the Waymo Open Dataset, Perception, v1.x tfrecords
(individual_files/validation/segment-*.tfrecord). The output is byte-identical to the reference
reference pixels when the chain below is followed exactly.

Reference chain (how the reference pixels were produced):
  Frame.images[<camera>].image (baseline JPEG, quality 90, 4:2:0, 1920x1280 or 1920x886)
    -> libjpeg decode with default settings (ISLOW IDCT, fancy chroma upsampling)
    -> no resize, no crop
    -> JPEG re-encode: quality 95, 4:2:0, baseline, standard Huffman tables (Pillow defaults)
Frame selection: release frame i (rgb/<cam>/%010d.jpg) = tfrecord record with index
identity['frames'][i]['source_frame_index'] (0-based position inside the segment file).

Dependencies: numpy + Pillow only. TFRecord framing and the protobuf wire format are parsed
here (no tensorflow, no waymo-open-dataset package); only the Frame fields that matter are read.

Identity (schema avb_restricted_identity_v1; one JSON per episode, or the fetch_manifest v2 block):
  schema           'avb_restricted_identity_v1'
  episode / split  release episode name and split (e.g. waymo_FL_ep000 / waymo_FL)
  dataset          'waymo'
  source           name ('waymo_open_dataset_perception_v1'), split ('validation'),
                   tfrecord (file name of the segment), context_name (Frame.context.name),
                   sorted_index (position of the file in the Python-sorted validation file list;
                   equals the val__scene index of the source data, informational)
  views            per release camera ('cam0' target, 'cam1' input): role, source_camera (Waymo
                   CameraName, e.g. FRONT_LEFT), camera_id (enum value 1..5), resolution [H, W]
  num_frames       41
  frames           num_frames entries: frame (release index), source_frame_index (record index
                   in the tfrecord), timestamp_micros (Frame.timestamp_micros, null if unknown)
  processing       the chain above, as data
  source_ids       source identifiers (SourceData path_sequence, camera folders, frame stems)
  sha256           optional, per view: {'%010d.jpg': sha256 of the reference pixels}
'''

import argparse
import hashlib
import io
import json
import os
import struct

import numpy as np
from PIL import Image

from anyview.restricted import compare_sha256

IDENTITY_SCHEMA = 'avb_restricted_identity_v1'
SOURCE_NAME = 'waymo_open_dataset_perception_v1'
FRAME_NAME = '{:010d}'

# dataset.proto CameraName.Name enum.
CAMERA_NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
CAMERA_IDS = {name: cid for (cid, name) in CAMERA_NAMES.items()}

# Reference JPEG encoder settings (Pillow; subsampling 2 = 4:2:0).
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 2

PROCESSING = {
    'decode': 'libjpeg defaults (ISLOW IDCT, fancy upsampling); Pillow and OpenCV agree',
    'resize': None,
    'crop': None,
    'encode': {'format': 'JPEG', 'quality': JPEG_QUALITY, 'subsampling': '4:2:0',
               'progressive': False, 'optimize': False, 'huffman': 'standard'},
}

# Waymo camera frame (x forward, y left, z up) -> release camera frame (x right, y down,
# z forward). Rows = release axes expressed in Waymo camera axes.
WAYMO_CAM_TO_RELEASE_CAM = np.array([[0.0, -1.0, 0.0],
                                     [0.0, 0.0, -1.0],
                                     [1.0, 0.0, 0.0]], dtype=np.float64)


# ---------------------------------------------------------------- TFRecord + protobuf wire


def iter_tfrecord(path):
    '''
    Yield (record_index, record_bytes) for a TFRecord file. The CRC32C footers are not
    checked (no dependency); record lengths are validated against the file size.
    '''
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        index = 0
        while True:
            header = f.read(12)
            if len(header) == 0:
                return
            if len(header) != 12:
                raise ValueError(f'{path}: truncated TFRecord header at record {index}')
            (length,) = struct.unpack('<Q', header[:8])
            if length > size:
                raise ValueError(f'{path}: bad record length {length} at record {index}')
            data = f.read(length)
            footer = f.read(4)
            if len(data) != length or len(footer) != 4:
                raise ValueError(f'{path}: truncated TFRecord record {index}')
            yield (index, data)
            index += 1


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            return result, pos


def _iter_fields(buf):
    '''
    Yield (field_number, wire_type, value) over one protobuf message. Values: int for varints,
    raw 8-byte / 4-byte strings for fixed64 / fixed32, bytes for length-delimited fields.
    '''
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire} (field {field})')
        yield (field, wire, value)


def _doubles(fields, number):
    '''
    Collect a repeated double field (packed or not) as a float64 array.
    '''
    values = []
    for (field, wire, value) in fields:
        if field != number:
            continue
        if wire == 1:
            values.append(struct.unpack('<d', value)[0])
        elif wire == 2:
            values.extend(np.frombuffer(value, dtype='<f8').tolist())
        else:
            raise ValueError(f'field {number}: unexpected wire type {wire} for double')
    result = np.asarray(values, dtype=np.float64)
    return result


def _parse_transform(buf):
    '''
    dataset.proto Transform (16 doubles, row-major 4x4).
    '''
    values = _doubles(list(_iter_fields(buf)), 1)
    if values.shape != (16,):
        raise ValueError(f'Transform with {values.shape} values (expected 16)')
    matrix = values.reshape(4, 4)
    return matrix


def _parse_calibration(buf):
    '''
    dataset.proto CameraCalibration: name (1), intrinsic (2, 9 doubles: f_u f_v c_u c_v k1 k2
    p1 p2 k3), extrinsic (3, camera -> vehicle), width (4), height (5).
    '''
    fields = list(_iter_fields(buf))
    calib = {'name': None, 'intrinsic': None, 'extrinsic': None, 'width': None, 'height': None}
    for (field, wire, value) in fields:
        if field == 1:
            calib['name'] = int(value)
        elif field == 3:
            calib['extrinsic'] = _parse_transform(value)
        elif field == 4:
            calib['width'] = int(value)
        elif field == 5:
            calib['height'] = int(value)
    calib['intrinsic'] = _doubles(fields, 2)
    return calib


def parse_frame(record, camera_ids=None):
    '''
    Parse the parts of a dataset.proto Frame that this module needs.
    :param camera_ids: iterable of CameraName values whose JPEG bytes to keep (None = all).
    :return dict: context_name (str), timestamp_micros (int), pose (4, 4) vehicle -> world,
        calibrations {camera_id: calib dict}, images {camera_id: jpeg bytes}.
    '''
    keep = None if camera_ids is None else set(int(c) for c in camera_ids)
    frame = {'context_name': None, 'timestamp_micros': None, 'pose': None,
             'calibrations': {}, 'images': {}}
    for (field, wire, value) in _iter_fields(record):
        if field == 1:  # Context
            for (cfield, cwire, cvalue) in _iter_fields(value):
                if cfield == 1:
                    frame['context_name'] = cvalue.decode('utf-8')
                elif cfield == 2:
                    calib = _parse_calibration(cvalue)
                    frame['calibrations'][calib['name']] = calib
        elif field == 2:
            frame['timestamp_micros'] = int(value)
        elif field == 3:
            frame['pose'] = _parse_transform(value)
        elif field == 4:  # CameraImage
            name = None
            jpeg = None
            for (ifield, iwire, ivalue) in _iter_fields(value):
                if ifield == 1:
                    name = int(ivalue)
                elif ifield == 2:
                    jpeg = ivalue
            if name is None or jpeg is None:
                raise ValueError('CameraImage without name or image bytes')
            if keep is None or name in keep:
                frame['images'][name] = bytes(jpeg)
        # Fields 5+ (lasers, labels, maps) are skipped.
    if frame['context_name'] is None or frame['timestamp_micros'] is None:
        raise ValueError('Frame without context name or timestamp')
    return frame


# ---------------------------------------------------------------- pixels


def reencode_reference_jpeg(jpeg_bytes):
    '''
    Official JPEG bytes -> reference JPEG bytes (decode, re-encode at quality 95, 4:2:0).
    '''
    with Image.open(io.BytesIO(jpeg_bytes)) as image:
        rgb = image.convert('RGB')
    buffer = io.BytesIO()
    rgb.save(buffer, format='JPEG', quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    result = buffer.getvalue()
    return result


def sha256_bytes(data):
    digest = hashlib.sha256(data).hexdigest()
    return digest


def release_cam2world(vehicle_pose, cam_extrinsic):
    '''
    Official vehicle -> world pose and camera -> vehicle extrinsic -> release cam2world (4, 4).
    '''
    axis = np.eye(4, dtype=np.float64)
    axis[:3, :3] = WAYMO_CAM_TO_RELEASE_CAM
    cam2world = vehicle_pose @ cam_extrinsic @ np.linalg.inv(axis)
    return cam2world


# ---------------------------------------------------------------- identity + lookup


def load_identity(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # fetch_manifest.json v2 wraps the identity block and keeps the sha256 table at the top level.
    identity = data.get('identity', data)
    if 'sha256' in data:
        identity.setdefault('sha256', data['sha256'])
    validate_identity(identity)
    return identity


def validate_identity(identity):
    if identity.get('schema') != IDENTITY_SCHEMA:
        raise ValueError(f'unsupported identity schema {identity.get("schema")!r}')
    if identity.get('dataset') != 'waymo':
        raise ValueError(f'identity is for dataset {identity.get("dataset")!r}; this extractor handles waymo')
    for key in ('episode', 'source', 'views', 'frames'):
        if key not in identity:
            raise KeyError(f'identity {identity.get("episode")}: missing {key!r}')
    for key in ('tfrecord', 'context_name'):
        if key not in identity['source']:
            raise KeyError(f'identity {identity["episode"]}: missing source.{key}')
    for cam in ('cam0', 'cam1'):
        view = identity['views'].get(cam)
        if view is None or view.get('source_camera') not in CAMERA_IDS:
            raise ValueError(f'identity {identity["episode"]}: bad view {cam}: {view}')
    for (i, frame) in enumerate(identity['frames']):
        if frame.get('frame') != i or 'source_frame_index' not in frame:
            raise ValueError(f'identity {identity["episode"]}: bad frames entry {i}: {frame}')
    num_frames = identity.get('num_frames')
    if num_frames is not None and int(num_frames) != len(identity['frames']):
        raise ValueError(f'identity {identity["episode"]}: num_frames {num_frames} != '
                         f'{len(identity["frames"])} frames entries')


def find_tfrecord(source_root, tfrecord_name):
    '''
    Locate one segment file under the official download root (flat, split subfolder, or the
    individual_files/<split> layout of the GCS bucket); falls back to a recursive search.
    '''
    candidates = [
        os.path.join(source_root, tfrecord_name),
        os.path.join(source_root, 'validation', tfrecord_name),
        os.path.join(source_root, 'individual_files', 'validation', tfrecord_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    for (dirpath, _, filenames) in os.walk(source_root):
        if tfrecord_name in filenames:
            return os.path.join(dirpath, tfrecord_name)
    raise FileNotFoundError(f'{tfrecord_name} not found under {source_root} (tried {candidates} '
                            f'and a recursive search)')


def read_frames(tfrecord_path, frame_indices, camera_ids):
    '''
    Parse the requested records (0-based indices) of one segment file.
    :return dict: {record_index: parsed frame}.
    '''
    wanted = set(int(i) for i in frame_indices)
    last = max(wanted)
    frames = {}
    for (index, record) in iter_tfrecord(tfrecord_path):
        if index in wanted:
            frames[index] = parse_frame(record, camera_ids)
        if index >= last:
            break
    missing = sorted(wanted - set(frames))
    if missing:
        raise ValueError(f'{tfrecord_path}: records {missing} not found (file has fewer frames)')
    return frames


def _check_lowdim(episode_dir, cam, frame_index, frame, camera_id, atol_px=0.05, atol_m=0.05):
    '''
    Compare the official calibration + vehicle pose with the episode's released lowdim (which is
    always available, also in the public tier). Catches a wrong segment or frame index early.
    '''
    path = os.path.join(episode_dir, 'lowdim', cam, FRAME_NAME.format(frame_index) + '.npz')
    if not os.path.isfile(path):
        return False
    data = np.load(path)
    K = np.asarray(data['intrinsics'], dtype=np.float64)
    cam2world = np.asarray(data['extrinsics'], dtype=np.float64)
    calib = frame['calibrations'].get(camera_id)
    if calib is None or calib['extrinsic'] is None or frame['pose'] is None:
        raise ValueError(f'{frame["context_name"]}: no calibration for camera {camera_id}')
    (f_u, f_v, c_u, c_v) = calib['intrinsic'][:4]
    expected_K = np.array([[f_u, 0.0, c_u], [0.0, f_v, c_v], [0.0, 0.0, 1.0]])
    if not np.allclose(K[:3, :3], expected_K, atol=atol_px):
        raise ValueError(f'{episode_dir} {cam} frame {frame_index}: intrinsics mismatch '
                         f'(released {K[:3, :3].tolist()} vs official {expected_K.tolist()})')
    expected = release_cam2world(frame['pose'], calib['extrinsic'])
    if not (np.allclose(cam2world[:3, :3], expected[:3, :3], atol=1e-3)
            and np.allclose(cam2world[:3, 3], expected[:3, 3], atol=atol_m)):
        raise ValueError(f'{episode_dir} {cam} frame {frame_index}: extrinsics mismatch (wrong '
                         f'segment or frame index?)\nreleased:\n{cam2world}\nofficial:\n{expected}')
    return True


def rebuild_episode(episode_dir, source_root, identity, check_lowdim=True, overwrite=False,
                        observed=None):
    '''
    Write rgb/cam0 and rgb/cam1 of one restricted episode from the official download.
    :param episode_dir: release episode directory (metadata.json + lowdim/ present).
    :param source_root: root of the official Waymo Open Dataset v1.x download.
    :param identity: identity dict (see module docstring) or path to its JSON file.
    :param check_lowdim: verify official calibration + pose against the released lowdim.
    :param observed: optional dict, filled with context_name and timestamp_micros (list per
        release frame) as read from the tfrecord.
    :return dict: {'rgb/cam0': {filename: sha256}, 'rgb/cam1': {...}} of the produced files.
    '''
    if isinstance(identity, str):
        identity = load_identity(identity)
    else:
        validate_identity(identity)

    views = identity['views']
    camera_ids = {cam: CAMERA_IDS[views[cam]['source_camera']] for cam in ('cam0', 'cam1')}
    frame_entries = identity['frames']
    source_indices = [int(entry['source_frame_index']) for entry in frame_entries]

    tfrecord_path = find_tfrecord(source_root, identity['source']['tfrecord'])
    frames = read_frames(tfrecord_path, source_indices, camera_ids.values())

    if observed is None:
        observed = {}
    observed['context_name'] = None
    observed['timestamp_micros'] = []
    result = {}
    for cam in ('cam0', 'cam1'):
        result[f'rgb/{cam}'] = {}
        os.makedirs(os.path.join(episode_dir, 'rgb', cam), exist_ok=True)

    for entry in frame_entries:
        i = int(entry['frame'])
        frame = frames[int(entry['source_frame_index'])]
        if frame['context_name'] != identity['source']['context_name']:
            raise ValueError(f'{tfrecord_path}: context name {frame["context_name"]} != identity '
                             f'{identity["source"]["context_name"]}')
        observed['context_name'] = frame['context_name']
        expected_ts = entry.get('timestamp_micros')
        if expected_ts is not None and int(expected_ts) != frame['timestamp_micros']:
            raise ValueError(f'{identity["episode"]} frame {i}: timestamp {frame["timestamp_micros"]} '
                             f'!= identity {expected_ts}')
        observed['timestamp_micros'].append(frame['timestamp_micros'])

        for cam in ('cam0', 'cam1'):
            camera_id = camera_ids[cam]
            if camera_id not in frame['images']:
                raise ValueError(f'{tfrecord_path} record {entry["source_frame_index"]}: no image '
                                 f'for camera {views[cam]["source_camera"]}')
            if check_lowdim:
                _check_lowdim(episode_dir, cam, i, frame, camera_id)
            name = FRAME_NAME.format(i) + '.jpg'
            out_path = os.path.join(episode_dir, 'rgb', cam, name)
            if os.path.isfile(out_path) and not overwrite:
                with open(out_path, 'rb') as f:
                    data = f.read()
            else:
                data = reencode_reference_jpeg(frame['images'][camera_id])
                with Image.open(io.BytesIO(data)) as image:
                    hw = [image.size[1], image.size[0]]
                expected_hw = views[cam].get('resolution')
                if expected_hw is not None and list(expected_hw) != hw:
                    raise ValueError(f'{identity["episode"]} {cam}: official image is {hw}, '
                                     f'identity expects {list(expected_hw)}')
                with open(out_path, 'wb') as f:
                    f.write(data)
            result[f'rgb/{cam}'][name] = sha256_bytes(data)

    return result


def main():
    ap = argparse.ArgumentParser(description='Rebuild one Waymo AnyViewBench episode.')
    ap.add_argument('--episode-dir', required=True)
    ap.add_argument('--source-root', required=True, help='official Waymo Open Dataset v1.x root')
    ap.add_argument('--identity', default=None,
                    help='identity / fetch manifest JSON (default: <episode-dir>/fetch_manifest.json)')
    ap.add_argument('--no-lowdim-check', action='store_true')
    args = ap.parse_args()
    identity_path = args.identity or os.path.join(args.episode_dir, 'fetch_manifest.json')
    identity = load_identity(identity_path)
    observed = {}
    produced = rebuild_episode(args.episode_dir, args.source_root, identity,
                                   check_lowdim=not args.no_lowdim_check, observed=observed)
    n_files = sum(len(v) for v in produced.values())
    print(f'{args.episode_dir}: {n_files} frames written from {observed["context_name"]} '
          f'(timestamps {observed["timestamp_micros"][0]}..{observed["timestamp_micros"][-1]})')
    if 'sha256' in identity:
        bad = compare_sha256(produced, identity['sha256'])
        n_expected = sum(len(v) for v in identity['sha256'].values())
        print(f'{n_expected - len(bad)}/{n_expected} sha256 match' + (f'; MISMATCH: {bad[:5]}' if bad else ''))


if __name__ == '__main__':
    main()
