# Ego-Exo4D restricted-split extractor: official downscaled 448p takes -> AVB reference JPEGs.
'''
Reproduces rgb/cam1 (input view) and rgb/cam0 (target view) of an Ego-Exo4D AnyViewBench episode
from the user's OFFICIAL Ego-Exo4D download, byte-identical to our reference pixels.

Reference chain (recovered from the original processing, 2026-09): the official downscaled take
videos takes/<take>/frame_aligned_videos/downscaled/448/<cam>.mp4 (h264, 30 fps, exo GoPro
796x448) were decoded with ffmpeg and every frame was written by ffmpeg's mjpeg encoder at
qscale 2 (libavcodec 58.91.100, FFmpeg 4.3). No resize, crop or rotation. An episode is 41
consecutive frames of two exo cameras (frame n of the video == take frame index n, frame-aligned
across cameras). ffmpeg's h264 decoder and mjpeg encoder are bit-exact across the versions we
tested (4.3, 4.4.2, 7.0.2); the only version-dependent bytes are the JPEG comment marker, which
this module rewrites to the reference value so SHA-256 checks pass.

Expected official layout under source_root (what the ego4d CLI writes for
`egoexo -o <source_root> --parts downscaled_takes/448 --uids <take_uid>`):
  <source_root>/takes/<take_name>/frame_aligned_videos/downscaled/448/<cam>.mp4
Only this downscaled part is needed (a few MB to about 100 MB per take; the full-resolution videos are unnecessary).

Identity fields used (the fetch_manifest.json v2 block, or an identity JSON of schema
avb_restricted_identity_v1): source.take_name, views.<cam>.source_camera, views.<cam>.video
(relative path under source_root; optional), first_frame_index (or frames[*].source_frame_index),
num_frames, and optionally sha256 for verification.
'''

import argparse
import hashlib
import json
import os
import shutil
import subprocess

from anyview.restricted import compare_sha256
from anyview.unified import FRAME_NAME

# COM marker payload of the reference JPEGs (libavcodec version string, NUL-terminated).
REFERENCE_JPEG_COMMENT = b'Lavc58.91.100\x00'
# mjpeg qscale used by the reference processing (ffmpeg -q:v 2).
REFERENCE_QSCALE = 2
VIDEO_RELPATH = 'takes/{take}/frame_aligned_videos/downscaled/448/{cam}.mp4'
DEFAULT_NUM_FRAMES = 41


def find_ffmpeg():
    '''
    ffmpeg binary: the imageio-ffmpeg wheel (pinned in requirements.txt) or a system ffmpeg.
    '''
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which('ffmpeg')
    if exe is None:
        raise RuntimeError('No ffmpeg found: pip install imageio-ffmpeg, or put ffmpeg on PATH')
    return exe


def set_jpeg_comment(data, comment):
    '''
    Replace the payload of the first COM (FFFE) segment of a JPEG (insert one after SOI when
    absent). Entropy-coded data is untouched.
    '''
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG (missing SOI)')
    segment = b'\xff\xfe' + (len(comment) + 2).to_bytes(2, 'big') + comment
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2:i + 4], 'big')
        if marker == 0xFE:
            patched = data[:i] + segment + data[i + 2 + length:]
            return patched
        if marker == 0xDA:  # SOS: no COM in the header
            break
        i += 2 + length
    patched = data[:2] + segment + data[2:]
    return patched


def extract_frames(video_path, first_index, num_frames, out_dir, ffmpeg=None):
    '''
    Decode frames [first_index, first_index + num_frames) of video_path and write them as JPEGs
    (mjpeg qscale 2, exactly like the reference processing) named FRAME_NAME(0..num_frames-1).
    '''
    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Missing official video: {video_path}')
    os.makedirs(out_dir, exist_ok=True)
    last_index = first_index + num_frames - 1
    # select by decoded frame number n (exact, independent of timestamps); image2 muxer writes
    # the selected frames in order without duplication (passthrough is its default).
    assert FRAME_NAME.format(7) == '%010d' % 7  # ffmpeg pattern below == unified frame stems
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-v', 'error', '-y',
           '-i', video_path,
           '-vf', f"select='between(n,{first_index},{last_index})'",
           '-frames:v', str(num_frames),
           '-q:v', str(REFERENCE_QSCALE),
           '-start_number', '0',
           os.path.join(out_dir, '%010d.jpg')]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg failed ({proc.returncode}) on {video_path}:\n'
                           f'{proc.stderr.decode(errors="replace")}')
    names = sorted(n for n in os.listdir(out_dir) if n.endswith('.jpg'))
    expected = [FRAME_NAME.format(i) + '.jpg' for i in range(num_frames)]
    if names != expected:
        raise RuntimeError(f'{video_path}: expected frames {first_index}..{last_index} '
                           f'({num_frames} files), got {len(names)}; is the video complete?')
    for name in names:
        fp = os.path.join(out_dir, name)
        with open(fp, 'rb') as f:
            data = f.read()
        with open(fp, 'wb') as f:
            f.write(set_jpeg_comment(data, REFERENCE_JPEG_COMMENT))
    return [os.path.join(out_dir, n) for n in names]


def sha256_dir(folder):
    out = {}
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), 'rb') as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def frame_window(identity):
    '''
    (first_frame_index, num_frames) from either the explicit field or the per-frame list.
    '''
    num_frames = int(identity.get('num_frames', DEFAULT_NUM_FRAMES))
    if 'first_frame_index' in identity:
        first = int(identity['first_frame_index'])
    elif 'frames' in identity:
        indices = [int(fr['source_frame_index']) for fr in identity['frames']]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError('identity.frames are not consecutive take frames; unsupported')
        first = indices[0]
        num_frames = len(indices)
    else:
        raise KeyError('identity lacks first_frame_index / frames')
    return first, num_frames


def rebuild_episode(episode_dir, source_root, identity, ffmpeg=None):
    '''
    Write rgb/cam0 and rgb/cam1 of one episode from the official layout at source_root.
    Returns {'rgb/cam0': {filename: sha256}, 'rgb/cam1': {...}} of the produced files.
    '''
    if ffmpeg is None:
        ffmpeg = find_ffmpeg()
    take = identity['source']['take_name']
    first, num_frames = frame_window(identity)
    result = {}
    for cam in ('cam0', 'cam1'):
        view = identity['views'][cam]
        rel = view.get('video') or VIDEO_RELPATH.format(take=take, cam=view['source_camera'])
        video_path = os.path.join(source_root, rel)
        if not os.path.isfile(video_path) and os.path.isdir(os.path.join(source_root, take)):
            video_path = os.path.join(source_root, os.path.relpath(rel, 'takes'))  # source_root = takes/
        final_dir = os.path.join(episode_dir, 'rgb', cam)
        partial_dir = final_dir + '.partial'
        if os.path.isdir(partial_dir):
            shutil.rmtree(partial_dir)
        extract_frames(video_path, first, num_frames, partial_dir, ffmpeg=ffmpeg)
        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)
        os.replace(partial_dir, final_dir)
        result[f'rgb/{cam}'] = sha256_dir(final_dir)
    return result


def main():
    ap = argparse.ArgumentParser(description='Rebuild one Ego-Exo4D AnyViewBench episode.')
    ap.add_argument('--episode-dir', required=True)
    ap.add_argument('--source-root', required=True, help='official Ego-Exo4D download root')
    ap.add_argument('--identity', default=None,
                    help='identity / fetch manifest JSON (default: <episode-dir>/fetch_manifest.json)')
    args = ap.parse_args()
    identity_path = args.identity or os.path.join(args.episode_dir, 'fetch_manifest.json')
    with open(identity_path, 'r') as f:
        data = json.load(f)
    # fetch_manifest.json v2 wraps the identity block and keeps the sha256 table at the top level.
    identity = data.get('identity', data)
    identity.setdefault('sha256', data.get('sha256'))
    produced = rebuild_episode(args.episode_dir, args.source_root, identity)
    n_files = sum(len(v) for v in produced.values())
    if identity.get('sha256'):
        bad = compare_sha256(produced, identity['sha256'])
        n_expected = sum(len(v) for v in identity['sha256'].values())
        print(f'{args.episode_dir}: {n_files} frames written, '
              f'{n_expected - len(bad)}/{n_expected} sha256 match'
              + (f'; MISMATCH: {bad[:5]}' if bad else ''))
    else:
        print(f'{args.episode_dir}: {n_files} frames written (no sha256 in identity to verify)')


if __name__ == '__main__':
    main()
