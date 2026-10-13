"""独立扩散图片优化入口，复用本地检索目标并保留原实验接口。"""
import argparse
import json
from pathlib import Path

from src.paths import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group7', action='store_true', help='使用冻结的第七组源图和目标图')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--options', type=Path)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--optimization-config', type=Path)
    args = parser.parse_args()
    if args.group7:
        if args.source or args.target:
            parser.error('--group7 与显式输入图互斥')
        assets = PROJECT / 'src/output/nine_groups_attacker_history_joint/assets'
        args.source = assets / '9b5f4a94d9ab7f2e001498260243b3d91eedcd774939a4558c4f5786192e3a6a.jpg'
        args.target = assets / 'dd0abf8d2d52aa289346aa2e7219187cec901c901d4f3cce1ef3379691467256.jpg'
    if args.source is None or args.target is None:
        parser.error('必须给出 --group7 或同时指定 --source 与 --target')
    options = json.loads(args.options.read_text(encoding='utf-8')) if args.options else {}
    from src.vision.frequency_attack import generate_attack
    image, metadata = generate_attack(args.source, args.target, args.output,
        method='diffusion', steps=args.steps, device=args.device,
        optimization_config=args.optimization_config, diffusion_options=options)
    print(json.dumps({'image': str(image), 'status': metadata['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
