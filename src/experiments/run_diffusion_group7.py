"""串行运行第七组真实扩散优化与统一比较，并保存可续做的阶段记录。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from src.paths import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'src/output/diffusion_group7_20261011')
    parser.add_argument('--wait-for-download', action='store_true')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--options', type=Path, default=PROJECT / 'src/configs/diffusion_group7.json')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'pipeline.json'
    def record(stage, status, error=None):
        state_path.write_text(json.dumps({'stage': stage, 'status': status, 'error': error,
            'python': sys.executable, 'steps': args.steps, 'options': str(args.options.resolve()),
            'updated_unix': time.time(),
            'output': str(output)}, ensure_ascii=False, indent=2), encoding='utf-8')
    def run(module, arguments):
        subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m', module, *arguments],
                       cwd=PROJECT, check=True)
    try:
        record('model_download', 'running')
        if args.wait_for_download:
            deadline = time.time() + 3 * 3600
            while not (PROJECT / 'tmp/models/stable-diffusion-v1-5/download_provenance.json').is_file():
                if time.time() > deadline:
                    raise TimeoutError('等待模型下载超过三小时，请查看下载日志后续跑')
                time.sleep(20)
        else:
            run('src.experiments.download_diffusion_model', [])
        record('image_generation', 'running')
        attack = output / 'attack'
        if not (attack / 'metadata.json').is_file():
            run('src.experiments.diffusion_image_optimization', [
                '--group7', '--output', str(attack), '--steps', str(args.steps),
                '--options', str(args.options.resolve())])
        metadata = json.loads((attack / 'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('status') != 'success':
            raise RuntimeError('图片生成元数据未确认成功')
        record('comparison', 'running')
        comparison = output / 'comparison'
        if not (comparison / 'report.json').is_file():
            run('src.experiments.compare_diffusion_group7', [
                '--diffusion', str(attack / 'adversarial.png'),
                '--reconstruction', str(attack / 'reconstruction.png'),
                '--output', str(comparison), '--device', 'cuda'])
        report = json.loads((comparison / 'report.json').read_text(encoding='utf-8'))
        if report.get('status') != 'completed':
            raise RuntimeError('统一比较未确认完成')
        record('awaiting_visual_review_and_user_delivery', 'computed')
        print('真实图片与指标已生成，等待检查图片并向公子展示', flush=True)
    except Exception as error:
        record('pipeline', 'failed', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
