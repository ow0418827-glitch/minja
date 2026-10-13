"""Import the three audited SCAR images into copied SSA histories; never optimize."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from src.environment.local_webshop import fingerprint, read, write
from src.experiments import nine_group_experiment as driver
from src.experiments.reused_frequency_pipeline import DEFAULT_SOURCE, DEFAULT_SSA_ROOT, profile_arguments
from src.experiments.reuse_frequency_history import prepare_reused_history, verify_reused_artifacts
from src.experiments.visual_memory_experiment import pixel_audit
from src.vision.attack_completion import validate_attack_completion


DEFAULT_ROOT = driver.OUTPUT / 'nine_groups_old_b_scar_126_equal05_full_20261008'
DEFAULT_REPORT = driver.OUTPUT / 'vision_optimization_20261008_0200/postprocess_scale/report.json'
CORE_METADATA = {
    '01_camera': 'benchmark_refine_01_camera/refine/metadata.json',
    '02_chair': 'benchmark_refine_02_chair/refine/metadata.json',
    '06_jeans': 'benchmark_optimized/optimized/metadata.json',
}


def asset(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Precomputed asset is missing: ' + str(path))
    return {'path': str(path), 'sha256': driver.digest(path)}


def precomputed_manifest(source_root, report_path):
    """Bind the selected images to the actual A/old-B pair and upstream records."""
    source_root, report_path = Path(source_root).resolve(), Path(report_path).resolve()
    report = read(report_path)
    if report.get('status') != 'completed':
        raise ValueError('SCAR selection report is incomplete')
    plan = read(source_root / 'manifest.json')
    groups = {group['id']: group for group in plan['groups']}
    records = {}
    for row in report.get('groups', []):
        group_id = row['group']
        if group_id not in CORE_METADATA or group_id in records:
            raise ValueError('SCAR selection contains an unexpected or duplicate group')
        group = groups[group_id]
        source = asset(row['source'])
        target = asset(group['query_image'])
        selected = asset(row['selected']['image'])
        input_image = asset(row['input'])
        if (source['sha256'] != row['source_sha256']
                or target['sha256'] != row['query_image_sha256']
                or selected['sha256'] != row['selected']['image_sha256']
                or input_image['sha256'] != row['input_sha256']):
            raise ValueError('SCAR selection image differs from its recorded digest')
        binding = read(source_root / 'groups' / group_id / 'history_binding.json')
        if (source['sha256'] != binding['source_image_sha256']
                or target['sha256'] != binding['reference_query_sha256']):
            raise ValueError('SCAR image does not match the actual source history and old B query')
        core = asset(report_path.parent.parent / CORE_METADATA[group_id])
        core_record = read(core['path'])
        records[group_id] = {
            'group': group_id, 'source': source, 'target': target,
            'selected': selected, 'input': input_image, 'report': asset(report_path),
            'core_metadata': core, 'selected_scale': row['selected']['scale'],
            'upstream_optimization_steps': core_record['attack']['steps_completed'],
            'upstream_best_step': core_record['attack']['best_step'],
            'upstream_selected_step': core_record['attack'].get('selected_step'),
            'selection_weights': row['criteria'],
            'evaluation_weights': {'key_image_weight': 0.5, 'query_image_weight': 0.5},
        }
    if set(records) != set(CORE_METADATA):
        raise ValueError('SCAR import requires the camera, chair and jeans selected images')
    return {'schema_version': 1, 'method': 'precomputed_scar_import',
            'source_root': str(source_root), 'groups': records}


def arguments(root=DEFAULT_ROOT, source_root=DEFAULT_SSA_ROOT, report=DEFAULT_REPORT,
              *, port=3007, device='cuda', condition=None):
    # This factory verifies the original fixed policy/template profile. The
    # history cloning source is explicitly the later SSA run, not that factory run.
    args = profile_arguments(DEFAULT_SOURCE, root, method='ssa', attack_steps=500,
                             port=port, device=device, index_text='full',
                             key_image_weight=0.5, query_image_weight=0.5)
    previous = read(Path(source_root) / 'run_settings.json')
    if any(previous.get(key) != value for key, value in (
            ('attack_method', 'ssa'), ('attack_steps', 500), ('memory_top_k', 1),
            ('memory_mode', 'full'), ('model', driver.load_model_config(args.llm_config).public()),
            ('num_steps', args.num_steps), ('memory_prompt_budget', args.memory_prompt_budget),
            ('policy_max_tokens', args.policy_max_tokens))):
        raise ValueError('The source SSA run does not match the controlled shopping profile')
    args.source_root = Path(source_root).resolve()
    args.method, args.attack_steps = 'scar', 0
    for name in ('ssa_samples', 'ssa_rho', 'ssa_sigma', 'ssa_momentum', 'ssa_options'):
        setattr(args, name, None)
    args.precomputed_manifest = precomputed_manifest(args.source_root, report)
    args.condition = condition
    driver.evaluation_conditions(args)
    return args


def import_images(args):
    """Copy selected bytes and create a truthful zero-new-step completion record."""
    root, plan, groups = driver.load_plan(args)
    manifest_path = root / 'scar_import_manifest.json'
    if manifest_path.exists() and read(manifest_path) != args.precomputed_manifest:
        raise ValueError('Frozen SCAR import manifest changed')
    write(manifest_path, args.precomputed_manifest)
    for prepared in groups:
        group = driver.resolve_history_binding(args, root, plan, prepared, required=True)
        folder = root / 'groups' / group['id']
        ready_path = folder / 'attack_ready.json'
        expected = driver.attack_settings(args, group)
        if ready_path.exists():
            ready = read(ready_path)
            if (ready['settings'] != expected or driver.digest(ready['image']) != ready['sha256']
                    or ready.get('history_binding_sha256') != group['history_binding_sha256']):
                raise ValueError('Existing imported SCAR image or history binding changed')
            validate_attack_completion(ready['metadata'], 0, expected)
            continue
        record = args.precomputed_manifest['groups'][group['id']]
        output = folder / 'attack' / 'adversarial.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if driver.digest(output) != record['selected']['sha256']:
                raise ValueError('Refusing to overwrite a different imported image')
        else:
            shutil.copy2(record['selected']['path'], output)
        audit = pixel_audit(group['source_image'], output, args.epsilon_levels)
        metadata = {
            'schema_version': 1, 'method': 'scar', 'generation': 'precomputed_scar_import',
            'status': 'success',
            'attack': {'steps_requested': 0, 'steps_completed': 0, 'best_step': 0,
                       'selected_step': 0, 'stop_reason': 'precomputed_import_completed'},
            'precomputed_import': {
                'manifest_sha256': fingerprint(args.precomputed_manifest),
                'group_record': record, 'output': asset(output), 'pixel_audit': audit},
            'source_image_sha256': expected['source_sha256'],
            'target_image_sha256': expected['target_sha256'],
            'output_image': str(output.resolve()), 'output_image_sha256': driver.digest(output),
        }
        validate_attack_completion(metadata, 0, expected)
        write(output.parent / 'metadata.json', metadata)
        write(ready_path, {'settings': expected, 'image': str(output.resolve()),
                          'sha256': driver.digest(output), 'pixel_audit': audit,
                          'metadata': metadata, 'history_binding_sha256': group['history_binding_sha256']})
        print(group['id'] + '：最终推荐图已复制并通过来源及像素预算校验。', flush=True)


def prepare(args):
    from src.experiments.prime_reused_retrieval import prime_retrieval
    provenance = prepare_reused_history(args.source_root, args)
    import_images(args)
    driver.finalize(args)
    audit = verify_reused_artifacts(args, provenance, require_banks=True)
    retrieval = prime_retrieval(args, provenance)
    driver.check(args)
    result = {'status': 'ready_for_shopping', 'method': 'scar', 'root': str(args.root),
              'new_optimization_steps': 0, 'remote_policy_calls': 0,
              'index_text': 'full', 'key_image_weight': 0.5, 'query_image_weight': 0.5,
              'formal_tests_per_condition_per_group': 20,
              'conditions': list(driver.evaluation_conditions(args)),
              'planned_total_shopping_episodes': 60 * len(driver.evaluation_conditions(args)),
              'audit': audit,
              'top1_counts': {g['group']: g['source_top1_counts'] for g in retrieval['groups']}}
    write(args.root / 'scar_preparation_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def audit_prepared(args):
    root, plan, _ = driver.load_plan(args)
    args.reuse_frozen_memory = read(root / 'run_settings.json') != driver.execution_settings(args, plan)
    if read(root / 'scar_import_manifest.json') != args.precomputed_manifest:
        raise ValueError('SCAR import manifest differs from the frozen experiment')
    provenance = read(root / 'history_reuse_provenance.json')
    verify_reused_artifacts(args, provenance, require_banks=True,
                            evaluation_only=args.reuse_frozen_memory)
    if read(root / 'scar_preparation_result.json').get('status') != 'ready_for_shopping':
        raise ValueError('SCAR images and retrieval vectors must be prepared before shopping')
    return root, provenance


def evaluate(args):
    root, provenance = audit_prepared(args)
    resume_source = getattr(args, 'reuse_results_manifest', None)
    if resume_source is not None:
        from src.experiments.reuse_condition_results import seed_selected_condition
        inherited = seed_selected_condition(args, read(resume_source))
        write(root / 'condition_resume_preparation.json', inherited)
        print('已核验并继承所选条件的已保存结果：'
              + str(inherited['inherited_total']) + ' 条。', flush=True)
    selected = driver.evaluation_conditions(args)
    print('已核验三组复制记忆及新检索向量，开始默认等权、完整历史的购物评估；运行条件：'
          + ','.join(selected), flush=True)
    with driver.running_server(args):
        driver.evaluate(args)
    verify_reused_artifacts(args, provenance, require_banks=True,
                            evaluation_only=args.reuse_frozen_memory)
    result = driver.status(args)
    write(root / 'pipeline_result.json', result)
    if getattr(args, 'condition', None) is not None:
        write(root / ('pipeline_' + args.condition + '_result.json'), result)
    print(json.dumps({'status': result['status'], 'root': str(root)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = ['prepare', 'evaluate', 'resume', 'run', 'check', 'status']
    parser.add_argument('stage', nargs='?', choices=stages)
    parser.add_argument('--stage', dest='stage_option', choices=stages)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SSA_ROOT)
    parser.add_argument('--selection-report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--port', type=int, default=3007)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--condition', choices=['all', 'adversarial', 'clean', 'no_memory'],
                        help='Run one memory condition, or all three; defaults to the original three-condition run')
    parser.add_argument('--reuse-results-manifest', type=Path,
                        help='Explicit audited snapshot of committed results to inherit for the selected conditions')
    cli = parser.parse_args()
    if cli.stage and cli.stage_option and cli.stage != cli.stage_option:
        parser.error('Stage arguments must agree')
    if not 1 <= cli.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    os.environ.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,' + os.environ.get('NO_PROXY', '')
    args = arguments(cli.root, cli.source_root, cli.selection_report, port=cli.port, device=cli.device,
                     condition=cli.condition)
    args.reuse_results_manifest = cli.reuse_results_manifest
    stage = cli.stage_option or cli.stage or 'run'
    if stage in ('prepare', 'run'):
        prepare(args)
    if stage in ('evaluate', 'resume', 'run'):
        evaluate(args)
    elif stage == 'check':
        audit_prepared(args)
        driver.check(args)
    elif stage == 'status':
        audit_prepared(args)
        print(json.dumps(driver.status(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
