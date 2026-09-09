# finetuning of AnyView-DVS on unified Kubric-5D scenes (torch DDP, bf16 autocast).
'''
Finetune the released 1->1 dynamic view synthesis model on the Kubric-5D training scenes
(unified scene layout, videos or frames variant). Plain torch DistributedDataParallel;
one process per GPU via torchrun.

Recipe (ported from the training code used for the paper): rectified-flow denoising with the release
pipeline's preconditioning; per-sample noise level sigma = exp(N(0, 1)) * sqrt(T_latent),
replaced with probability 0.05 by a log-uniform draw in [200, 1e5]; the same sigma for both
view streams; conditioning latents get uniform[0.001, 0.01] Gaussian augmentation; loss =
mean over the target-view rgb latent of (sigma^2 + 1) / sigma^2 * (x0_pred - x0)^2, times
loss_scale 10 / 4; AdamW(betas 0.9/0.99, weight decay 0.1) with linear warmup then linear
decay to 5 percent; activation checkpointing 'mm_only'; fp32 master weights, bf16 compute.

Example (2 GPUs):
    torchrun --nproc_per_node=2 scripts/train_dvs.py --data-root /data/Kubric5D \
        --ckpt checkpoints/anyview_dvs_2b.pt --tokenizer checkpoints/tokenizer.pth \
        --out outputs/finetune --max-steps 500
'''

import argparse
import json
import math
import os
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_RELEASE_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import infer as infer_lib


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune AnyView-DVS on unified Kubric-5D scenes',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, required=True,
                        help='Directory of unified scenes (scnNNNNN/metadata.json + rgb/ + lowdim/)')
    parser.add_argument('--ckpt', type=str, required=True, help='Warm-start checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, required=True, help='Video tokenizer (.pth)')
    parser.add_argument('--out', type=str, required=True, help='Output directory (checkpoints + log)')
    parser.add_argument('--max-steps', type=int, default=500, help='Optimizer steps to run')
    parser.add_argument('--batch-size', type=int, default=1, help='Per-GPU batch size')
    parser.add_argument('--num-frames', type=int, default=41, help='Clip length (1 + 4k)')
    parser.add_argument('--resolution', type=int, default=576,
                        help='Long side of the training clips (576 = the released model; '
                             '384 or 320 when memory is tight)')
    parser.add_argument('--lr', type=float, default=2e-6, help='Peak learning rate (AdamW)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Linear warmup steps; then linear decay to 5 percent at --max-steps')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='AdamW weight decay')
    parser.add_argument('--sac-mode', type=str, default='mm_only',
                        help="Activation checkpointing: 'mm_only' (training default) or 'none'")
    parser.add_argument('--save-every', type=int, default=250,
                        help='Checkpoint interval in steps (bf16, anyview_dvs_step{N:06d}.pt; also at the end)')
    parser.add_argument('--log-every', type=int, default=10, help='Log interval in steps (train_log.jsonl)')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers per process')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed (rank is added)')
    args = parser.parse_args()
    return args


def setup_distributed():
    '''
    torchrun environment -> (rank, world_size, local_rank); single process without torchrun.
    '''
    import torch
    import torch.distributed as dist

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', device_id=torch.device('cuda', local_rank))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        (rank, world_size, local_rank) = (0, 1, 0)
    torch.cuda.set_device(local_rank)
    return (rank, world_size, local_rank)


def lr_lambda_factory(warmup_steps, max_steps, f_start=0.01, f_min=0.05):
    '''
    Linear warmup from f_start to 1, then linear decay to f_min at max_steps.
    '''
    def lr_lambda(step):
        if step < warmup_steps:
            frac = f_start + (1.0 - f_start) * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            frac = 1.0 + (f_min - 1.0) * min(1.0, progress)
        return frac
    return lr_lambda


def sample_sigmas(batch_size, latent_frames, device, high_sigma_ratio=0.05):
    '''
    Training noise levels, one per sample (shared by both view streams).
    '''
    import torch

    log_sigma = torch.randn(batch_size, device=device, dtype=torch.float64)
    sigma = torch.exp(log_sigma) * math.sqrt(latent_frames)
    high = torch.rand(batch_size, device=device, dtype=torch.float64) < high_sigma_ratio
    log_uniform = torch.rand(batch_size, device=device, dtype=torch.float64) \
        * (math.log(1e5) - math.log(200.0)) + math.log(200.0)
    sigma = torch.where(high, torch.exp(log_uniform), sigma)
    sigma = sigma.to(torch.float32).view(batch_size, 1, 1, 1, 1)
    return sigma


def prepare_batch(items, vae, config, resolution, device):
    '''
    Raw dataset samples -> entries (latents + constant DVS masks) for a batch of one or more
    clips at the snapped training resolution. Every clip in a batch shares one grid.
    '''
    import torch
    from anyview.logistics import build_dvs_entries

    (rgb0_list, rgb1_list, cams0_list, cams1_list) = ([], [], [], [])
    for item in items:
        input_frames = item['input'].numpy()
        target_frames = item['target_gt'].numpy()
        hw1n = tuple(input_frames.shape[1:3])
        hw0n = tuple(target_frames.shape[1:3])
        hw1 = infer_lib.snap_shape(hw1n[0], hw1n[1], target=resolution)
        hw0 = infer_lib.snap_shape(hw0n[0], hw0n[1], target=resolution)

        cams = {
            'intrinsics': {0: item['intrinsics_target'].float(), 1: item['intrinsics_input'].float()},
            'world2cam': {0: item['world2cam_target'].float(), 1: item['world2cam_input'].float()},
        }
        cams['world2cam'] = infer_lib.reanchor_world2cam(cams['world2cam'])
        scale_factor = float(item['scale_factor'])

        with torch.no_grad():
            video1 = infer_lib.resize_video(input_frames, hw1)
            video0 = infer_lib.resize_video(target_frames, hw0)
            rgb1_list.append(vae.encode_rgb(video1.unsqueeze(0).to(device)))
            rgb0_list.append(vae.encode_rgb(video0.unsqueeze(0).to(device)))
            cams0_list.append(infer_lib.prepare_cams_latent(
                vae, cams['intrinsics'][0], cams['world2cam'][0], hw0n, hw0, scale_factor, device))
            cams1_list.append(infer_lib.prepare_cams_latent(
                vae, cams['intrinsics'][1], cams['world2cam'][1], hw1n, hw1, scale_factor, device))

    entries = build_dvs_entries(torch.cat(rgb0_list), torch.cat(rgb1_list),
                                torch.cat(cams0_list), torch.cat(cams1_list))
    return entries


def training_loss(pipe, entries, sigma_data=1.0, cond_aug_range=(0.001, 0.01),
                  loss_scale=10.0):
    '''
    One denoising step with the release preconditioning; EDM-weighted masked MSE on the
    target-view rgb latent (the only supervised entry under the constant DVS masks).
    '''
    import torch
    from anyview.logistics import pack_streams_from_entries, unpack_entries_from_streams

    (x0_streams, masks) = pack_streams_from_entries(entries)
    x0_streams = {k: v.float() for (k, v) in x0_streams.items()}
    B = x0_streams['v0'].shape[0]
    device = x0_streams['v0'].device
    latent_frames = x0_streams['v0'].shape[2]

    sigma = sample_sigmas(B, latent_frames, device)
    sigmas = {k: sigma for k in x0_streams}
    yt_streams = {k: v + sigmas[k] * torch.randn_like(v) for (k, v) in x0_streams.items()}

    # Conditioning augmentation on the clean streams that feed the network as inputs.
    (lo, hi) = cond_aug_range
    ca = (lo + (hi - lo) * torch.rand(B, device=device)).view(B, 1, 1, 1, 1)
    x0_noised = {k: v + ca * torch.randn_like(v) for (k, v) in x0_streams.items()}

    crossattn_emb = pipe.empty_text_emb.expand(B, -1, -1)
    (_, y0_pred_streams) = pipe.denoise(x0_noised, yt_streams, masks, sigmas, crossattn_emb)

    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    losses = {}
    for k in x0_streams:
        mse = (y0_pred_streams[k].float() - x0_streams[k]) ** 2
        losses[k] = mse * masks['supervise'][k].float() * weight
    loss_entries = unpack_entries_from_streams(losses)
    # Under the constant DVS masks only rgb0 carries supervision; the training code used for the paper sums
    # the active per-entry means and divides by 4 before applying loss_scale.
    loss = loss_entries['rgb0'].mean() / 4.0 * loss_scale
    return loss


def save_checkpoint(dit_module, path, step):
    '''
    Flat bf16 state dict under the net. prefix (same size and layout as the released
    checkpoint), loadable by anyview.pipe.load_pipeline.
    '''
    import torch
    from anyview.pipe import SAC_WRAP

    state = {'net.' + SAC_WRAP.sub('', k): v.detach().to('cpu', torch.bfloat16)
             for (k, v) in dit_module.state_dict().items()}
    torch.save({'model': state, 'step': step}, path)


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    from anyview.config import AnyViewConfig
    from anyview.kubric_dataset import KubricDVSDataset
    from anyview.pipe import AnyViewPipeline, load_dit, load_text_emb
    from anyview.vae import load_vae

    (rank, world_size, local_rank) = setup_distributed()
    device = torch.device('cuda', local_rank)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f'world_size={world_size} batch/gpu={args.batch_size} steps={args.max_steps} '
              f'res={args.resolution} sac={args.sac_mode}')

    config = AnyViewConfig(checkpoint_path=args.ckpt, tokenizer_path=args.tokenizer)
    vae = load_vae(config.tokenizer_path, device)
    dit = load_dit(config, sac_mode=args.sac_mode).to(device=device, dtype=torch.float32).train()
    dit_ddp = dit
    if world_size > 1:
        # The released checkpoint carries parameters this task never touches (view-0 timestep
        # embedder, rope buffers), so DDP must not wait for their gradients.
        dit_ddp = DistributedDataParallel(dit, device_ids=[local_rank], find_unused_parameters=True)
    # The pipeline casts network inputs to bf16; fp32 master weights compute under autocast.
    pipe = AnyViewPipeline(dit_ddp, load_text_emb(config), device=str(device), dtype=torch.bfloat16)

    dataset = KubricDVSDataset(args.data_root, num_frames=args.num_frames, seed=args.seed + rank)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                     seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers,
                        collate_fn=lambda items: items, drop_last=False, persistent_workers=False)
    if len(loader) == 0:
        raise SystemExit(f'rank {rank}: no batches (dataset of {len(dataset)} scenes is too small for '
                         f'{world_size} processes x batch {args.batch_size})')

    optimizer = torch.optim.AdamW(dit.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                  weight_decay=args.weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup_steps, args.max_steps))

    log_path = os.path.join(args.out, 'train_log.jsonl')
    step = 0
    epoch = 0
    t_last = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for items in loader:
            if step >= args.max_steps:
                break
            entries = prepare_batch(items, vae, config, args.resolution, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = training_loss(pipe, entries)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if is_main and (step % args.log_every == 0 or step == 1):
                now = time.time()
                peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                record = {'step': step, 'loss': float(loss.item()),
                          'lr': float(scheduler.get_last_lr()[0]),
                          'sec_per_step': (now - t_last) / (args.log_every if step > 1 else 1),
                          'peak_vram_gb': round(peak_gb, 2)}
                print(json.dumps(record))
                with open(log_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                t_last = now
            if is_main and (step % args.save_every == 0 or step == args.max_steps):
                path = os.path.join(args.out, f'anyview_dvs_step{step:06d}.pt')
                save_checkpoint(dit, path, step)
                print(f'saved {path}')
        epoch += 1

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
