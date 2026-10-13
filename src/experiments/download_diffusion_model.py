"""下载独立扩散实验所需权重，默认优先使用中国大陆镜像。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/models/stable-diffusion-v1-5'))
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    import requests
    repository = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    response = requests.get(f'{args.endpoint}/api/models/{repository}/revision/{args.revision}', timeout=30)
    response.raise_for_status()
    revision = response.json()['sha']
    output = args.output.resolve()
    patterns = ['model_index.json', 'README.md', 'LICENSE*',
                'scheduler/*', 'tokenizer/*', 'unet/config.json', 'vae/config.json',
                'text_encoder/config.json', 'unet/*fp16.safetensors',
                'vae/*fp16.safetensors', 'text_encoder/*fp16.safetensors']
    print(f'下载固定版本 {revision} 到 {output}', flush=True)
    response = requests.get(f'{args.endpoint}/api/models/{repository}/tree/{revision}',
                            params={'recursive': 'true'}, timeout=30)
    response.raise_for_status()
    files = [item for item in response.json() if item.get('type') == 'file'
             and any(fnmatch.fnmatch(item['path'], pattern) for pattern in patterns)]
    def digest(path):
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(block)
        return sha.hexdigest()
    def download(item):
        destination = output / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('oid')
        if destination.is_file() and destination.stat().st_size == item['size']:
            if expected is None or digest(destination) == expected:
                print(f"已有校验文件 {item['path']}", flush=True)
                return
        partial = destination.with_suffix(destination.suffix + '.partial')
        url = f"{args.endpoint}/{repository}/resolve/{revision}/{item['path']}"
        print(f"下载 {item['path']}，{item['size'] / 1024**2:.1f} 兆字节", flush=True)
        if item['size'] > 16 * 1024**2:
            chunk_size = 8 * 1024**2
            pieces = destination.parent / (destination.name + '.parts')
            pieces.mkdir(exist_ok=True)
            def fetch(start):
                end = min(item['size'], start + chunk_size) - 1
                piece = pieces / str(start)
                if piece.is_file() and piece.stat().st_size == end - start + 1:
                    return piece
                for attempt in range(4):
                    try:
                        resource = requests.get(url + '?download=true',
                            headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 90))
                        resource.raise_for_status()
                        if (resource.status_code != 206 or
                                resource.headers.get('Content-Range') != f"bytes {start}-{end}/{item['size']}" or
                                len(resource.content) != end - start + 1):
                            raise RuntimeError('分段响应内容或范围不匹配')
                        temp = piece.with_suffix('.partial')
                        temp.write_bytes(resource.content)
                        temp.replace(piece)
                        if start // chunk_size % 20 == 0:
                            print(f"分段完成 {item['path']} {start / 1024**2:.0f} 兆字节", flush=True)
                        return piece
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 * (attempt + 1))
            with ThreadPoolExecutor(max_workers=args.workers) as segments:
                ordered = list(segments.map(fetch, range(0, item['size'], chunk_size)))
            with partial.open('wb') as stream:
                for piece in ordered:
                    stream.write(piece.read_bytes())
        else:
            resource = requests.get(url, timeout=(30, 90))
            resource.raise_for_status()
            partial.write_bytes(resource.content)
        if partial.stat().st_size != item['size'] or (expected and digest(partial) != expected):
            raise RuntimeError(f"下载文件校验失败：{item['path']}")
        partial.replace(destination)
        print(f"完成 {item['path']}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))
    (output / 'download_provenance.json').write_text(json.dumps({
        'repository': repository, 'revision': revision, 'endpoint': args.endpoint,
        'variant': 'fp16', 'allow_patterns': patterns,
        'files': [{'path': item['path'], 'size': item['size'],
                   'sha256': digest(output / item['path'])} for item in files],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('权重下载完成', flush=True)


if __name__ == '__main__':
    main()
