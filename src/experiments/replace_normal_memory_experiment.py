"""Replace one ordinary jeans history, preserving the SCAR image and twenty queries."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from src.environment.local_webshop import read, write, fingerprint
from src.experiments import nine_group_experiment as driver
from src.experiments.reuse_frequency_history import _copy_resources, _copy_exact, _relocate, DEMAND_FIELDS
from src.experiments.reused_scar_pipeline import arguments as scar_arguments, import_images
from src.experiments.visual_memory_experiment import eligible
from src.vision.joint_encoder import make_joint_encoder
from src.vision.visual_retrieval import VisualMemoryRetriever

SOURCE = driver.OUTPUT / 'nine_groups_old_b_scar_126_equal05_full_20261008'
ROOT = driver.OUTPUT / 'scar_jeans_replace_normal_20261008'
GROUP = '06_jeans'
REMOVE_ID = 'fixed_nine_p6_normal_16_0_2554'
SOURCE_VARIANT = '416d088c36098ba4_level_1_1a2a1db1f40c1464'


def arguments(root=ROOT, source=SOURCE, *, port=3007, device='cuda'):
    args = scar_arguments(root, port=port, device=device, condition='adversarial')
    args.group, args.source_root = ['6'], Path(source).resolve()
    args.history_reserve_count = 5
    args.normal_history_candidate_policy = 'one_primary_then_five_frozen_reserves_for_replacement_v1'
    source_state = read(args.source_root / 'groups' / GROUP / 'history_state.json')
    old_plan = read(args.source_root / 'manifest.json')
    group = next(group for group in old_plan['groups'] if group['id'] == GROUP)
    removed = [row for row in source_state['normal'] if row['Id'] == REMOVE_ID]
    if len(removed) != 1 or removed[0]['FinalASIN'] != 'B081YYST69':
        raise ValueError('Source normal memory does not match the requested removal')
    used_indexes = {row['candidate_index'] for row in source_state['attempts']
                    if row.get('kind') == 'normal' and row.get('candidate_index') is not None}
    excluded = {task['task_id'] for index, task in enumerate(group['normal_candidates'])
                if index in used_indexes or index < source_state['cursor']}
    selection = driver.history_candidate_plan(group, primary_count=1, reserve_count=5,
                                               exclude_task_ids=excluded)
    source_folder = args.source_root / 'groups' / GROUP
    paths = [args.source_root / 'manifest.json', args.source_root / 'run_settings.json']
    paths += [source_folder / name for name in ('history_state.json', 'history_memory.json',
               'history_binding.json', 'memory.json', 'clean_memory.json', 'memory_ready.json',
               'attack_ready.json')]
    paths += [path for path in (source_folder / 'evaluations' / SOURCE_VARIANT).rglob('*') if path.is_file()]
    policy = {'version': 1, 'source_root': str(args.source_root), 'group': GROUP,
              'removed_memory_id': REMOVE_ID, 'removed_asin': removed[0]['FinalASIN'],
              'removed_position': source_state['normal'].index(removed[0]),
              'retained_normal_ids': [row['Id'] for row in source_state['normal'] if row['Id'] != REMOVE_ID],
              'retained_normal_sha256': {row['Id']: fingerprint(row) for row in source_state['normal']
                                         if row['Id'] != REMOVE_ID},
              'designated_history_sha256': fingerprint(source_state['designated']),
              'source_files': {str(path.resolve()): driver.digest(path) for path in paths},
              'candidate_plan': selection, 'selection': 'first_valid_purchase_without_rank_filter',
              'excluded_replacement_asins': [removed[0]['FinalASIN']],
              'existing_normal_count': 13, 'required_new_normal_count': 1,
              'normal_retries': args.normal_retries,
              'runner_sha256': driver.digest(Path(__file__))}
    args.precomputed_manifest['normal_replacement_policy'] = policy
    args.replacement_policy = policy
    return args


def verify_source(policy):
    for name, expected in policy['source_files'].items():
        if driver.digest(name) != expected:
            raise ValueError('Original experiment source changed: ' + name)


def clone(args):
    root = driver.workspace_output(args.root)
    policy = args.replacement_policy
    verify_source(policy)
    marker = root / 'normal_replacement_policy.json'
    if root.exists():
        if not marker.is_file() or read(marker) != policy:
            raise ValueError('Refusing to overwrite an unrelated replacement experiment')
        _, plan, _ = driver.load_plan(args)
        driver.freeze_settings(args, root, plan)
        return
    source = args.source_root
    original_plan = read(source / 'manifest.json')
    original_group = next(g for g in original_plan['groups'] if g['id'] == GROUP)
    driver.verify_collected_history(args, source, original_plan, original_group)
    driver.verify_banks(source / 'groups' / GROUP)
    root.mkdir(parents=True)
    write(marker, policy)
    _copy_resources(source, root)
    source_folder, folder = source / 'groups' / GROUP, root / 'groups' / GROUP
    for name in ('memory_images', 'images'):
        for path in sorted((source_folder / name).rglob('*')):
            if path.is_file():
                _copy_exact(path, folder / path.relative_to(source_folder))
    plan = _relocate(copy.deepcopy(original_plan), source, root)
    plan['bundle_id'] = fingerprint({key: value for key, value in plan.items() if key != 'bundle_id'})
    write(root / 'manifest.json', plan)
    write(root / 'task_plan.json', _relocate(read(source / 'task_plan.json'), source, root))
    group = next(g for g in plan['groups'] if g['id'] == GROUP)
    write(folder / 'tasks.json', group)
    state = copy.deepcopy(read(source_folder / 'history_state.json'))
    removed = state['normal'].pop(policy['removed_position'])
    assert removed['Id'] == REMOVE_ID
    state['normal_collection_plan'] = policy['candidate_plan']
    state['normal_collection_audit'] = driver.history_reserve_audit(
        policy['candidate_plan'], state['attempts'], normal_retries=args.normal_retries)
    write(folder / 'history_state.json', state)
    write(folder / 'history_memory.json', state['normal'] + [state['designated']])
    _, checked_plan, _ = driver.load_plan(args)
    driver.freeze_settings(args, root, checked_plan)


def collect_replacement(args):
    root, plan, groups = driver.load_plan(args)
    group, policy = groups[0], args.replacement_policy
    folder = root / 'groups' / GROUP
    if (folder / 'history_binding.json').exists():
        driver.verify_collected_history(args, root, plan, group)
        return
    state_path, memory_path = folder / 'history_state.json', folder / 'history_memory.json'
    state = read(state_path)
    if state['normal_collection_plan'] != policy['candidate_plan']:
        raise ValueError('Replacement candidate plan changed')
    for attempt in state['attempts']:
        if attempt.get('status') == 'running':
            attempt.update(status='interrupted', accepted=False, completed=False,
                           technical_error=True, error='InterruptedAttempt', reward=None)
    write(state_path, state)
    rt = driver.runtime(args)
    rt['auxiliary'] = driver.load_auxiliary(driver.DEFAULT_TEMPLATE)[0]
    excluded_asins = {g['target_asin'] for g in plan['groups']} | set(policy['excluded_replacement_asins'])
    try:
        for entry in policy['candidate_plan']['candidates']:
            if len(state['normal']) == 14:
                break
            index, item = entry['candidate_index'], entry['task']
            previous = [a for a in state['attempts'] if a.get('kind') == 'normal'
                        and a.get('candidate_index') == index]
            for retry in range(len(previous), args.normal_retries):
                task = dict(item, session=f"fixed_nine_p6_normal_{index}_{retry}_{item['goal_index']}")
                print(f"替换正常记忆：{entry['role']}，候选{index + 1}，尝试{retry + 1}/{args.normal_retries}。", flush=True)
                accepted = driver.collect_history_attempt(args, root, group, task, state, state_path,
                    memory_path, rt, excluded_asins, kind='normal', candidate_index=index)
                if accepted:
                    break
        if len(state['normal']) != 14:
            write(root / 'replacement_failure.json', state['normal_collection_audit'])
            raise RuntimeError('Replacement primary and five reserves exhausted; no memory was fabricated')
    finally:
        close = getattr(rt.get('visual_encoder'), 'close', None)
        if callable(close):
            close()
    new_rows = [row for row in state['normal'] if row['Id'] not in policy['retained_normal_ids']]
    if len(new_rows) != 1 or new_rows[0]['FinalASIN'] in excluded_asins:
        raise ValueError('The replacement purchase does not satisfy the requested exclusion')
    replacement = new_rows[0]
    state['normal'] = [row for row in state['normal'] if row['Id'] != replacement['Id']]
    state['normal'].insert(policy['removed_position'], replacement)
    state['normal_collection_audit'] = driver.history_reserve_audit(
        policy['candidate_plan'], state['attempts'], normal_retries=args.normal_retries)
    write(state_path, state)
    write(memory_path, state['normal'] + [state['designated']])
    driver.save_history_binding(args, root, plan, group, state)
    print('新的真实购买记忆已生成，并放回被替换记忆的原位置。', flush=True)


def verify_replacement(args):
    root, plan, groups = driver.load_plan(args)
    policy = args.replacement_policy
    verify_source(policy)
    if read(root / 'normal_replacement_policy.json') != policy:
        raise ValueError('Frozen replacement policy changed')
    if read(root / 'run_settings.json') != driver.execution_settings(args, plan):
        raise ValueError('Replacement run settings or code changed')
    group = groups[0]
    folder = root / 'groups' / GROUP
    driver.verify_collected_history(args, root, plan, group)
    state = read(folder / 'history_state.json')
    if fingerprint(state['designated']) != policy['designated_history_sha256']:
        raise ValueError('Designated attack history changed')
    retained = [row for row in state['normal'] if row['Id'] in policy['retained_normal_ids']]
    if len(retained) != 13 or any(fingerprint(row) != policy['retained_normal_sha256'][row['Id']] for row in retained):
        raise ValueError('One of the other thirteen normal memories changed')
    replacement = [row for row in state['normal'] if row['Id'] not in policy['retained_normal_ids']]
    if (len(replacement) != 1 or any(row['Id'] == REMOVE_ID for row in state['normal'])
            or state['normal'][policy['removed_position']]['Id'] != replacement[0]['Id']):
        raise ValueError('Exactly the requested normal slot must change')
    ready = driver.verify_banks(folder)
    old_folder = args.source_root / 'groups' / GROUP
    old_ready = driver.verify_banks(old_folder)
    if ready['attack_sha256'] != old_ready['attack_sha256']:
        raise ValueError('The SCAR attack image changed during normal replacement')
    old_bank, new_bank = read(old_folder / 'memory.json'), read(folder / 'memory.json')
    old_by_id, new_by_id = {r['Id']: r for r in old_bank}, {r['Id']: r for r in new_bank}
    for memory_id in policy['retained_normal_ids']:
        if old_by_id[memory_id] != new_by_id[memory_id]:
            raise ValueError('Retained normal memory bank content changed')
    source_id = ready['source_memory_id']
    strip = lambda row: {k: v for k, v in row.items() if k not in ('RetrievalImage', 'ImageSubstitution')}
    if strip(old_by_id[source_id]) != strip(new_by_id[source_id]):
        raise ValueError('Designated attack memory facts changed')
    return {'source_files_unchanged': True, 'retained_normal_memories': 13,
            'removed_memory_id': REMOVE_ID, 'replacement_memory_id': replacement[0]['Id'],
            'replacement_product': replacement[0]['FinalASIN'],
            'attack_image_unchanged': True, 'normal_count': 14, 'memory_count': 15,
            'history_reserve_audit': state['normal_collection_audit']}


def prime(args):
    root, plan, groups = driver.load_plan(args)
    folder = root / 'groups' / GROUP
    group = driver.resolve_history_binding(args, root, plan, groups[0], required=True)
    location, _ = driver.evaluation_location(args, folder, save=True)
    tasks, tasks_sha = driver.evaluation_trials(args, group, folder, task_folder=location)
    source_trials = read(args.source_root / 'groups' / GROUP / 'evaluations' / SOURCE_VARIANT / 'targeted_tasks.json')['trials']
    if len(tasks) != 20 or any({k: a.get(k) for k in DEMAND_FIELDS} != {k: b.get(k) for k in DEMAND_FIELDS}
                             for a, b in zip(source_trials, tasks)):
        raise ValueError('Fixed original twenty test queries changed')
    encoder = make_joint_encoder(kind='clip', device=args.device, optimization_config=args.optimization_config)
    try:
        retriever = VisualMemoryRetriever(encoder, folder / 'memory.json', key_mode='joint', query_mode='joint',
            index_text='full', key_image_weight=0.5, query_image_weight=0.5)
        bank = eligible(read(folder / 'memory.json'))
        source_id = read(folder / 'memory_ready.json')['source_memory_id']
        rows = []
        for task in tasks:
            rankings = retriever.rank(bank, task['instruction'], group['query_image'])
            source = next(row for row in rankings if row['Id'] == source_id)
            rival = max((row for row in rankings if row['Id'] != source_id), key=lambda row: row['score'])
            rows.append({'task_id': task['task_id'], 'source_rank': source['rank'],
                         'source_score': source['score'], 'leading_margin': source['score'] - rival['score'],
                         'top_competitor_id': rival['Id']})
        report = {'schema_version': 1, 'group': GROUP, 'index_text': 'full',
                  'key_image_weight': 0.5, 'query_image_weight': 0.5, 'test_count': 20,
                  'same_twenty_queries': True, 'encoder': encoder.snapshot, 'retrieval_snapshot': retriever.snapshot,
                  'tasks_sha256': tasks_sha, 'rankings': rows,
                  'source_top1': sum(row['source_rank'] == 1 for row in rows),
                  'mean_margin': float(np.mean([row['leading_margin'] for row in rows])),
                  'min_margin': min(row['leading_margin'] for row in rows), 'remote_policy_calls': 0}
        write(root / 'replacement_retrieval_preparation.json', report)
        print(f"新记忆库离线检索：指定错误记忆首位{report['source_top1']}/20。", flush=True)
        return report
    finally:
        encoder.close()


def prepare(args):
    clone(args)
    with driver.running_server(args):
        collect_replacement(args)
    import_images(args)
    driver.finalize(args)
    audit = verify_replacement(args)
    retrieval = prime(args)
    driver.check(args)
    write(args.root / 'replacement_preparation_result.json', {'status': 'ready_for_evaluation',
        'audit': audit, 'retrieval': retrieval, 'requested_condition': 'adversarial',
        'groups': [GROUP], 'planned_shopping_episodes': 20, 'optimization_steps': 0})


def evaluate(args):
    verify_replacement(args)
    if read(args.root / 'replacement_preparation_result.json')['status'] != 'ready_for_evaluation':
        raise ValueError('Replacement and new retrieval index must be prepared first')
    print('开始牛仔裤组20条对抗图片条件评估；仅正常记忆替换，图片与查询保持原样。', flush=True)
    with driver.running_server(args):
        driver.evaluate(args)
    verify_replacement(args)
    write(args.root / 'pipeline_result.json', driver.status(args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'evaluate', 'run', 'check', 'status'], default='run')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--port', type=int, default=3007)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    cli = parser.parse_args()
    if not 1 <= cli.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    os.environ.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,' + os.environ.get('NO_PROXY', '')
    args = arguments(cli.root, cli.source_root, port=cli.port, device=cli.device)
    if cli.stage in ('prepare', 'run'):
        prepare(args)
    if cli.stage in ('evaluate', 'run'):
        evaluate(args)
    elif cli.stage == 'check':
        verify_replacement(args)
        driver.check(args)
    elif cli.stage == 'status':
        verify_replacement(args)
        print(json.dumps(driver.status(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
