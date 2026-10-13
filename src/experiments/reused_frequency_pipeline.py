"""Run old-B comparisons with selectable low-frequency or SSA images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver


DEFAULT_SOURCE = driver.OUTPUT / 'nine_groups_vague_top1_batch_126'
DEFAULT_ROOT = driver.OUTPUT / 'nine_groups_old_b_frequency_126_500steps_20261006'
DEFAULT_SSA_ROOT = driver.OUTPUT / 'nine_groups_old_b_ssa_126_500steps_20261006'


def profile_arguments(source_root, root, *, port=3007, device='cuda',
                      method='frequency_momentum', attack_steps=500,
                      frequency_axis_ratio=None, frequency_momentum=None,
                      ssa_samples=None, ssa_rho=None, ssa_sigma=None, ssa_momentum=None,
                      ssa_options=None, index_text='full', key_image_weight=0.5, query_image_weight=0.5,
                      condition=None):
    """Keep the source purchase/retrieval settings and change only the attack."""
    source_root = Path(source_root).resolve()
    if method not in ('frequency_momentum', 'ssa'):
        raise ValueError('Old-B comparison requires frequency_momentum or ssa')
    if isinstance(attack_steps, bool) or not isinstance(attack_steps, int) or attack_steps < 1:
        raise ValueError('Attack steps must be a positive integer')
    from src.experiments.visual_memory_experiment import ssa_options as parse_ssa_options, retrieval_options
    selected = SimpleNamespace(method=method, ssa_samples=ssa_samples, ssa_rho=ssa_rho,
        ssa_sigma=ssa_sigma, ssa_momentum=ssa_momentum, ssa_options=ssa_options,
        index_text=index_text, key_image_weight=key_image_weight, query_image_weight=query_image_weight)
    spectrum_options = parse_ssa_options(selected)
    retrieval_options(selected)
    if method == 'ssa':
        if frequency_axis_ratio is not None or frequency_momentum is not None:
            raise ValueError('Frequency parameters require frequency_momentum')
        from src.vision.ssa_attack import ssa_settings
        ssa_settings(**spectrum_options)
    else:
        if any(value is not None for value in spectrum_options.values()):
            raise ValueError('SSA parameters require ssa')
        frequency_axis_ratio = 0.25 if frequency_axis_ratio is None else frequency_axis_ratio
        frequency_momentum = 1.0 if frequency_momentum is None else frequency_momentum
        from src.vision.frequency_attack import frequency_settings
        frequency_settings(frequency_axis_ratio=frequency_axis_ratio,
                           frequency_momentum=frequency_momentum)
    plan = read(source_root / 'manifest.json')
    previous = read(source_root / 'run_settings.json')
    args = SimpleNamespace(
        stage='run', source_root=source_root, root=Path(root).resolve(), catalog=Path(previous['catalog']),
        webshop_source=Path(plan['webshop_source']), pair_file=Path(plan['pair_file']),
        reference_overrides=None, llm_config=driver.DEFAULT_CONFIG,
        indication_prompt=driver.DEFAULT_TEMPLATE, indication_level=1,
        port=port, group=['1', '2', '6'], seed=plan['seed'],
        attack_steps=attack_steps, method=method, frequency_axis_ratio=frequency_axis_ratio,
        frequency_momentum=frequency_momentum, **spectrum_options,
        epsilon_levels=16, optimization_config=None, device=device,
        num_steps=previous['num_steps'], memory_top_k=1, memory_mode='full',
        memory_prompt_budget=previous['memory_prompt_budget'], query_image_mode='reference',
        policy_query_image=True, test_mode='targeted', query_specificity='vague', reserve_tests=0,
        retrieval_encoder='clip', memory_key_mode='joint', memory_query_mode='joint', index_text=index_text,
        key_image_weight=key_image_weight, query_image_weight=query_image_weight,
        gme_python=Path('D:/Anaconda/envs/dual-visual/python.exe'),
        gme_model='Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
        policy_max_tokens=previous['policy_max_tokens'], normal_candidate_limit=plan['normal_candidate_limit'],
        normal_retries=previous['normal_retries'], history_max_attempts=previous['history_max_attempts'],
        history_product_mode=previous['history_product_mode'], source_history_mode='attacker',
        include_controls=True, condition=condition, retry_errors=False, allow_code_update_resume=False,
        allow_bound_history_repair_resume=False, reuse_frozen_memory=False,
        model_check=False, direct_images=False)
    _, template = driver.load_auxiliary(args.indication_prompt, args.indication_level)
    driver.evaluation_conditions(args)
    model = driver.load_model_config(args.llm_config)
    if (previous.get('protocol') != driver.ATTACKER_PROTOCOL
            or previous.get('attack_method') != 'legacy_pixel' or previous.get('attack_steps') != 1000
            or previous.get('memory_top_k') != 1 or previous.get('memory_mode') != 'full'
            or previous.get('model') != model.public() or previous.get('template') != template
            or previous.get('epsilon_levels') != 16):
        raise ValueError('The old pixel comparison does not match the fixed model/template/top-one/budget profile')
    return args


def run_pipeline(args, *, prepare_history=None, verify_history=None, prime=None, image_only=False):
    """No recollection; rebuild images/banks/indices before any new shopping call."""
    if prepare_history is None or verify_history is None:
        from src.experiments.reuse_frequency_history import prepare_reused_history, verify_reused_artifacts
        prepare_history = prepare_history or prepare_reused_history
        verify_history = verify_history or verify_reused_artifacts
    if prime is None:
        from src.experiments.prime_reused_retrieval import prime_retrieval
        prime = prime_retrieval
    provenance = prepare_history(args.source_root, args)
    print(f'Reused old-B histories verified; generating three '
          f'{getattr(args, "attack_steps", 500)}-step {getattr(args, "method", "frequency_momentum")} attacks.',
          flush=True)
    driver.attack(args)
    driver.finalize(args)
    verify_history(args, provenance, require_banks=True)
    retrieval_report = prime(args, provenance)
    driver.check(args)
    if image_only:
        result = {'status': 'image_optimization_completed', 'image_only': True,
                  'root': str(args.root), 'method': getattr(args, 'method', 'frequency_momentum'),
                  'attack_steps_requested': getattr(args, 'attack_steps', 500),
                  'remote_policy_calls': 0,
                  'retrieval_report': str(args.root / 'retrieval_index_preparation.json'),
                  'groups': (retrieval_report or {}).get('groups', [])}
        write(args.root / 'image_optimization_result.json', result)
        print('图片优化、记忆索引和离线排名核验完成。', flush=True)
        return result
    print('New old-B memory/query vectors ready; starting three-condition shopping evaluation.', flush=True)
    with driver.running_server(args):
        driver.evaluate(args)
    verify_history(args, provenance, require_banks=True)
    result = driver.status(args)
    write(args.root / 'pipeline_result.json', result)
    print(json.dumps({'state': result['status'], 'formal_tests_per_condition': 20,
                      'groups': len(result['groups']), 'target_purchase_rate': result['target_purchase_rate']},
                     ensure_ascii=False), flush=True)
    return result


def resume_evaluation(args):
    """Resume shopping with frozen images, without recollection or optimization.

    A changed evaluation implementation uses its own protocol-scoped directory.
    Within that directory, committed demand/condition results are reused.
    Older protocol results and frozen preparation provenance are preserved;
    a matching unfinished run receives its final canonical pipeline summary.
    """
    args.stage = 'evaluate'
    root, plan, groups = driver.load_plan(args)
    # Matching code/settings must keep the original protocol directory. Turning
    # on frozen-memory reuse unconditionally would create another variant and
    # repeat already committed shopping results instead of resuming them.
    same_run = read(root / 'run_settings.json') == driver.execution_settings(args, plan)
    args.reuse_frozen_memory = not same_run
    if args.reuse_frozen_memory:
        driver.verify_frozen_history_for_evaluation(args, root, plan, groups)
    print('核验冻结图片和记忆完成；开始购物评估或断点续跑，不重新优化图片。', flush=True)
    with driver.running_server(args):
        driver.evaluate(args)
    result = driver.status(args)
    if same_run:
        write(root / 'pipeline_result.json', result)
    write(root / 'evaluation_resume_result.json', result)
    print(json.dumps({'state': result['status'], 'groups': len(result['groups']),
                      'executed_primary_tests': result['executed_primary_tests'],
                      'target_purchase_rate': result['target_purchase_rate']},
                     ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = ['prepare', 'optimize', 'run', 'evaluate', 'resume', 'check', 'status']
    parser.add_argument('stage', choices=stages, nargs='?')
    parser.add_argument('--stage', dest='stage_option', choices=stages,
                        help='optimize: images, banks and offline retrieval only')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--root', type=Path,
                        help='Separate output directory; defaults to a method-specific root')
    parser.add_argument('--port', type=int, default=3007)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--method', choices=['frequency_momentum', 'ssa'], default='frequency_momentum')
    parser.add_argument('--attack-steps', type=int, default=500)
    parser.add_argument('--frequency-axis-ratio', type=float)
    parser.add_argument('--frequency-momentum', type=float)
    parser.add_argument('--ssa-samples', type=int)
    parser.add_argument('--ssa-rho', type=float)
    parser.add_argument('--ssa-sigma', type=float)
    parser.add_argument('--ssa-momentum', type=float)
    parser.add_argument('--ssa-options', help='SSA early-stop/refinement JSON object or UTF-8 JSON file')
    parser.add_argument('--index-text', choices=['full', 'instruction', 'semantic'], default='full')
    parser.add_argument('--key-image-weight', type=float, default=0.5)
    parser.add_argument('--query-image-weight', type=float, default=0.5)
    parser.add_argument('--condition', choices=['all', 'adversarial', 'clean', 'no_memory'],
                        help='Select a single shopping condition, or all three; defaults preserve the original run')
    cli = parser.parse_args()
    if cli.stage_option and cli.stage and cli.stage_option != cli.stage:
        parser.error('Positional stage and --stage must agree')
    cli.stage = cli.stage_option or cli.stage or 'run'
    if not 1 <= cli.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if cli.attack_steps < 1:
        parser.error('Attack steps must be positive')
    cli.root = cli.root or (DEFAULT_SSA_ROOT if cli.method == 'ssa' else DEFAULT_ROOT)
    os.environ.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,' + os.environ.get('NO_PROXY', '')
    args = profile_arguments(cli.source_root, cli.root, port=cli.port, device=cli.device,
                             method=cli.method, attack_steps=cli.attack_steps,
                             frequency_axis_ratio=cli.frequency_axis_ratio,
                             frequency_momentum=cli.frequency_momentum,
                             ssa_samples=cli.ssa_samples, ssa_rho=cli.ssa_rho,
                             ssa_sigma=cli.ssa_sigma, ssa_momentum=cli.ssa_momentum,
                             ssa_options=cli.ssa_options, index_text=cli.index_text,
                             key_image_weight=cli.key_image_weight, query_image_weight=cli.query_image_weight,
                             condition=cli.condition)
    args.source_root = cli.source_root.resolve()
    if cli.stage == 'prepare':
        from src.experiments.reuse_frequency_history import prepare_reused_history
        provenance = prepare_reused_history(args.source_root, args)
        print(json.dumps({'root': str(args.root), 'groups': [1, 2, 6],
                          'state': 'histories_reused_ready_for_frequency_attack',
                          'method': args.method, 'attack_steps': args.attack_steps, 'remote_model_calls': 0},
                         ensure_ascii=False), flush=True)
    elif cli.stage in ('run', 'optimize'):
        run_pipeline(args, image_only=cli.stage == 'optimize')
    elif cli.stage in ('evaluate', 'resume'):
        resume_evaluation(args)
    elif cli.stage == 'check':
        driver.check(args)
    else:
        print(json.dumps(driver.status(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
