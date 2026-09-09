# PSNR / SSIM / LPIPS evaluation + AVB paper-table aggregation.
'''
Image metrics for AnyView-DVS evaluation.

PSNR and SSIM replicate skimage.metrics.peak_signal_noise_ratio and structural_similarity
(data_range=1.0, channel_axis=2, default 7x7 uniform window) in float64 numpy, so the numbers
match the original evaluation stack without a scikit-image dependency. LPIPS uses the lpips
package (VGG backbone). Aggregation helpers reproduce the paper tables: per-episode metrics,
per-split means, per-dataset means, and the in-distribution / zero-shot panel averages.
'''

import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib

METRIC_NAMES = ('PSNR', 'SSIM', 'LPIPS')

# Samples whose ground truth is entirely black are marked invalid with this value.
EMPTY_GT_SENTINEL = -999.0

# The 46 quant_v2 benchmark split names (datacam_name set of eval_mappings4.yaml).
SPLIT_NAMES = (
    'argoverse_FL', 'argoverse_FR', 'argoverse_LF', 'argoverse_RF',
    'assemblyhands',
    'ddad_FL', 'ddad_FR', 'ddad_LF', 'ddad_RF',
    'droid_ID_LR', 'droid_ID_RL', 'droid_OOD_LR', 'droid_OOD_RL',
    'dycheckm_5seq',
    'egoexo4d_ID_01', 'egoexo4d_ID_03', 'egoexo4d_ID_10', 'egoexo4d_ID_12',
    'egoexo4d_ID_21', 'egoexo4d_ID_23', 'egoexo4d_ID_30', 'egoexo4d_ID_32',
    'egoexo4d_OOD_01', 'egoexo4d_OOD_03', 'egoexo4d_OOD_10', 'egoexo4d_OOD_12',
    'egoexo4d_OOD_21', 'egoexo4d_OOD_23', 'egoexo4d_OOD_30', 'egoexo4d_OOD_32',
    'kubric4d_dir', 'kubric4d_grad', 'kubric5d',
    'lbmv12_LR', 'lbmv12_RL',
    'lyftl5_FL', 'lyftl5_FR', 'lyftl5_LF', 'lyftl5_RF',
    'pd4d', 'pd4d_dir', 'pd4d_grad',
    'waymo_FL', 'waymo_FR', 'waymo_LF', 'waymo_RF',
)

# Paper "extreme DVS" panels: canonical dataset -> its splits. Multi-camera datasets average
# their per-split means; each panel average is the unweighted mean over its datasets.
ID_PANEL = {
    'DROID': ('droid_ID_LR', 'droid_ID_RL'),
    'EgoExo4D': ('egoexo4d_ID_01', 'egoexo4d_ID_03', 'egoexo4d_ID_10', 'egoexo4d_ID_12',
                 'egoexo4d_ID_21', 'egoexo4d_ID_23', 'egoexo4d_ID_30', 'egoexo4d_ID_32'),
    'LBM': ('lbmv12_LR', 'lbmv12_RL'),
    'Kubric-4D': ('kubric4d_dir',),
    'Kubric-5D': ('kubric5d',),
    'Lyft': ('lyftl5_FL', 'lyftl5_FR', 'lyftl5_LF', 'lyftl5_RF'),
    'ParDom-4D': ('pd4d_dir',),
    'Waymo': ('waymo_FL', 'waymo_FR', 'waymo_LF', 'waymo_RF'),
}

ZERO_SHOT_PANEL = {
    'Argoverse': ('argoverse_FL', 'argoverse_FR', 'argoverse_LF', 'argoverse_RF'),
    'AssemblyHands': ('assemblyhands',),
    'DDAD': ('ddad_FL', 'ddad_FR', 'ddad_LF', 'ddad_RF'),
    'DROID': ('droid_OOD_LR', 'droid_OOD_RL'),
    'EgoExo4D': ('egoexo4d_OOD_01', 'egoexo4d_OOD_03', 'egoexo4d_OOD_10', 'egoexo4d_OOD_12',
                 'egoexo4d_OOD_21', 'egoexo4d_OOD_23', 'egoexo4d_OOD_30', 'egoexo4d_OOD_32'),
}

# Gradual-motion splits of the paper's narrow-DVS table (not part of either panel above).
GRADUAL_SPLITS = ('dycheckm_5seq', 'kubric4d_grad', 'pd4d_grad')


def _box_mean_valid(img, win):
    '''
    Valid-mode box-filter mean of a 2D float64 array via cumulative sums. Equals
    scipy.ndimage.uniform_filter cropped by (win - 1) // 2 per side, which is exactly the
    region SSIM averages over, so the boundary mode never enters the result.
    '''
    c = np.cumsum(np.cumsum(img, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    c = np.pad(c, ((1, 0), (1, 0)))
    sums = c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]
    means = sums / float(win * win)
    return means


class SSIM:
    '''
    structural_similarity replica: data_range=1.0, per-channel over channel_axis=2, uniform
    7x7 window, K1=0.01, K2=0.03, sample covariance; edge strip of (win - 1) // 2 cropped.
    Takes (C, H, W) tensors in [0, 1].
    '''

    def __init__(self):
        self.win_size = 7
        self.k1 = 0.01
        self.k2 = 0.03
        self.data_range = 1.0

    def __call__(self, pred, gt):
        pred_np = pred.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        gt_np = gt.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        win = self.win_size
        if min(pred_np.shape[0], pred_np.shape[1]) < win:
            raise ValueError(f'Image {pred_np.shape} smaller than SSIM window size {win}')
        num_px = win * win
        cov_norm = num_px / (num_px - 1)  # sample covariance
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        channel_vals = []
        for ch in range(pred_np.shape[2]):
            x = pred_np[:, :, ch]
            y = gt_np[:, :, ch]
            ux = _box_mean_valid(x, win)
            uy = _box_mean_valid(y, win)
            vx = cov_norm * (_box_mean_valid(x * x, win) - ux * ux)
            vy = cov_norm * (_box_mean_valid(y * y, win) - uy * uy)
            vxy = cov_norm * (_box_mean_valid(x * y, win) - ux * uy)
            a1 = 2.0 * ux * uy + c1
            a2 = 2.0 * vxy + c2
            b1 = ux * ux + uy * uy + c1
            b2 = vx + vy + c2
            ssim_map = (a1 * a2) / (b1 * b2)
            channel_vals.append(ssim_map.mean(dtype=np.float64))
        ssim = float(np.mean(channel_vals))
        return ssim


class PSNR:
    '''
    peak_signal_noise_ratio replica with data_range=1.0. Takes (C, H, W) tensors in [0, 1].
    '''

    def __call__(self, pred, gt):
        pred_np = pred.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        gt_np = gt.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        mse = np.mean((pred_np - gt_np) ** 2, dtype=np.float64)
        psnr = float(10.0 * np.log10(1.0 / mse))
        return psnr


class LPIPS:
    '''
    LPIPS (VGG backbone) via the lpips package; inputs in [0, 1] are mapped to [-1, 1].
    '''

    def __init__(self):
        self.criterion = lpips_lib.LPIPS(net='vgg', verbose=False)
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

    def __call__(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        self.criterion = self.criterion.to(pred.device).to(pred.dtype)
        val = self.criterion(pred * 2 - 1, gt * 2 - 1)
        return val


def _interpolate(tensor, size, scale_factor):
    # align_corners=True matches the original evaluation stack.
    out = F.interpolate(tensor, size=size, scale_factor=scale_factor,
                        recompute_scale_factor=False, mode='bilinear', align_corners=True)
    return out


class RGBEvaluation:
    '''
    PSNR / SSIM / LPIPS over a batch of images.

    resize = fixed (H, W) applied to pred AND gt before metrics (e.g. eval HR inference at LR
    metric resolution); resize_scale = aspect-preserving scale factor instead. crop_edges =
    fraction of height and width cropped from each border before metrics.
    '''

    def __init__(self, resize=None, resize_scale=None, crop_edges=None):
        self.metrics = METRIC_NAMES
        self.ssim = SSIM()
        self.psnr = PSNR()
        self.lpips = LPIPS()
        self.crop_edges = crop_edges
        if resize is not None and resize_scale is not None:
            raise ValueError('resize and resize_scale are mutually exclusive')
        if resize is not None:
            self.resize = partial(_interpolate, size=tuple(resize), scale_factor=None)
        elif resize_scale is not None:
            self.resize = partial(_interpolate, size=None, scale_factor=resize_scale)
        else:
            self.resize = None

    def compute(self, gt, pred):
        '''
        gt, pred: (B, C, H, W) in [0, 1]. Returns a (B, 3) tensor of [PSNR, SSIM, LPIPS] rows;
        samples with empty (all-black) ground truth get the -999 sentinel in all columns.
        '''
        metrics = []
        for pred_i, gt_i in zip(pred, gt):

            gt_i = gt_i.unsqueeze(0).clone().to(torch.float64)
            pred_i = pred_i.unsqueeze(0).clone().to(torch.float64)

            if self.resize is not None:
                gt_i = self.resize(gt_i)
                pred_i = self.resize(pred_i)

            gt_i = gt_i.clamp(min=0.0, max=1.0)
            pred_i = pred_i.clamp(min=0.0, max=1.0)

            if self.crop_edges:
                h, w = gt_i.shape[-2:]
                crop_h = int(self.crop_edges * h)
                crop_w = int(self.crop_edges * w)
                if crop_h > 0 and crop_w > 0:
                    gt_i = gt_i[:, :, crop_h:-crop_h, crop_w:-crop_w]
                    pred_i = pred_i[:, :, crop_h:-crop_h, crop_w:-crop_w]

            ssim = self.ssim(pred_i[0], gt_i[0])
            psnr = self.psnr(pred_i[0], gt_i[0])
            lpips_val = self.lpips(pred_i[0], gt_i[0])

            if gt_i.sum() < 0.01:
                psnr, ssim, lpips_val = -999, -999, -999

            metrics.append([psnr, ssim, lpips_val])

        result = torch.tensor(metrics, dtype=gt.dtype, device=gt.device)
        return result

    def evaluate(self, gt, pred):
        return self.compute(gt, pred)


def episode_split(episode):
    '''
    Maps an AVB episode name (<split>_ep<NNN>, e.g. waymo_FL_ep003) to its split name.
    '''
    if '_ep' not in episode:
        raise ValueError(f'Bad AVB episode name: {episode}')
    split = episode.rsplit('_ep', 1)[0]
    return split


def aggregate_per_split(per_episode):
    '''
    per_episode: {episode_name: {metric: value}} with AVB episode names. Returns
    {split: {metric: mean}}; episodes carrying the empty-GT sentinel are dropped.
    '''
    buckets = {}
    for episode, vals in per_episode.items():
        split = episode_split(episode)
        floats = {m: float(vals[m]) for m in METRIC_NAMES}
        if any(v <= EMPTY_GT_SENTINEL + 1.0 for v in floats.values()):
            continue
        buckets.setdefault(split, []).append(floats)
    per_split = {}
    for split, rows in buckets.items():
        per_split[split] = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
    return per_split


def aggregate_per_dataset(per_split, panel):
    '''
    per_split: {split: {metric: mean}}; panel: {dataset: (splits,)}. Returns
    {dataset: {metric: mean}} as the unweighted mean over the dataset's available splits;
    missing splits are warned and skipped, datasets with no splits are omitted.
    '''
    per_dataset = {}
    for dataset, splits in panel.items():
        rows = []
        for split in splits:
            if split not in per_split:
                warnings.warn(f'Missing split {split} for dataset {dataset}')
                continue
            rows.append(per_split[split])
        if not rows:
            continue
        per_dataset[dataset] = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
    return per_dataset


def aggregate_panels(per_split):
    '''
    Paper-table aggregation: per-dataset means, then the unweighted average over the datasets
    of each panel. Returns {'in_distribution': ..., 'zero_shot': ...}, each with 'datasets'
    ({dataset: {metric: mean}}) and 'mean' ({metric: mean}, or None if no data).
    '''
    panels = {}
    for name, panel in (('in_distribution', ID_PANEL), ('zero_shot', ZERO_SHOT_PANEL)):
        per_dataset = aggregate_per_dataset(per_split, panel)
        entry = {'datasets': per_dataset, 'mean': None}
        if per_dataset:
            rows = list(per_dataset.values())
            entry['mean'] = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
        panels[name] = entry
    return panels
