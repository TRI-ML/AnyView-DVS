# manifest-driven downloader for the public bucket (no listing permission needed).
'''
Download one release tier from its key manifest. The bucket allows reading individual keys;
listing is disabled, so every tier includes a manifest (one JSON line per file with key, size and
sha256). Downloads run in parallel over plain HTTPS, resume by skipping files whose size and
sha256 already match, and verify every file.

Tiers: Kubric5D = the Kubric-5D archives (train parts, val, test) with their .sha256 files and
Kubric5D_index.json; checkpoints = the model files. The benchmark archives are single files
(curl commands in the README).

Examples:
    python scripts/download.py --tier Kubric5D --out data
    python scripts/download.py --tier Kubric5D --out data --only 'Kubric5D_val*' 'Kubric5D_train_part0[0-2]*'
    python scripts/download.py --tier checkpoints --out checkpoints
    python scripts/download.py --manifest /path/to/manifest.jsonl --out /path/to/out
'''

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = 'https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com'
MANIFEST_PREFIX = 'datasets/anyview/manifests'
TIERS = ('Kubric5D', 'checkpoints')


def parse_args():
    parser = argparse.ArgumentParser(description='Download an AnyView-DVS release tier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tier', type=str, choices=TIERS, default=None,
                        help='Tier whose manifest is fetched from the bucket')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Local manifest path or URL (overrides --tier)')
    parser.add_argument('--out', type=str, required=True, help='Destination directory')
    parser.add_argument('--only', type=str, nargs='*', default=None,
                        help='Shell-style patterns on file names; only matching files are fetched '
                             "(e.g. 'Kubric5D_val*' 'Kubric5D_train_part0[0-2]*')")
    parser.add_argument('--workers', type=int, default=8, help='Parallel downloads')
    parser.add_argument('--base-url', type=str, default=BASE_URL)
    args = parser.parse_args()
    if args.tier is None and args.manifest is None:
        parser.error('pass --tier or --manifest')
    return args


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(source):
    if source.startswith('http://') or source.startswith('https://'):
        with urllib.request.urlopen(source) as r:
            text = r.read().decode('utf-8')
    else:
        with open(source, 'r') as f:
            text = f.read()
    entries = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return entries


def fetch(entry, base_url, out_root, prefix):
    '''
    Download one key to <out_root>/<key minus prefix>; skip when size + sha256 already match.
    '''
    rel = entry['key'][len(prefix) + 1:] if entry['key'].startswith(prefix + '/') else entry['key']
    dst = os.path.join(out_root, rel)
    if os.path.isfile(dst) and os.path.getsize(dst) == entry['size'] and sha256_of(dst) == entry['sha256']:
        return (rel, 'kept')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    with urllib.request.urlopen(f'{base_url}/{entry["key"]}') as r, open(tmp, 'wb') as f:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            f.write(chunk)
    if os.path.getsize(tmp) != entry['size'] or sha256_of(tmp) != entry['sha256']:
        os.remove(tmp)
        raise IOError(f'{entry["key"]}: size or sha256 mismatch after download')
    os.replace(tmp, dst)
    return (rel, 'downloaded')


def main():
    args = parse_args()
    source = args.manifest or f'{args.base_url}/{MANIFEST_PREFIX}/{args.tier}.jsonl'
    entries = read_manifest(source)
    if args.only:
        entries = [e for e in entries
                   if any(fnmatch.fnmatch(os.path.basename(e['key']), pat) for pat in args.only)]
    if not entries:
        raise SystemExit(f'no files selected from {source} (patterns: {args.only})')
    # Common key prefix = the tier folder on the bucket; files land relative to it.
    prefix = os.path.commonpath([e['key'] for e in entries]) if len(entries) > 1 \
        else os.path.dirname(entries[0]['key'])
    total = sum(e['size'] for e in entries)
    print(f'{len(entries)} files, {total / 1e9:.2f} GB -> {args.out} (from {source})')

    counts = {'kept': 0, 'downloaded': 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, e, args.base_url, args.out, prefix): e for e in entries}
        for (i, fut) in enumerate(as_completed(futures), start=1):
            entry = futures[fut]
            try:
                (rel, status) = fut.result()
                counts[status] += 1
            except Exception as e:
                failed.append(entry['key'])
                print(f'FAILED {entry["key"]}: {e}', file=sys.stderr)
            if i % 500 == 0 or i == len(entries):
                print(f'  {i}/{len(entries)} done ({counts["downloaded"]} downloaded, {counts["kept"]} kept)')
    if failed:
        raise SystemExit(f'{len(failed)} files failed; rerun to resume')
    print('all files verified')


if __name__ == '__main__':
    main()
