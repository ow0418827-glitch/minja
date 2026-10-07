"""The old-B comparison must never recollect or evaluate before rebuilding vectors."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.experiments import reused_frequency_pipeline as pipeline


def test_pipeline_rebuilds_only_attack_assets_before_new_evaluation(tmp_path, monkeypatch):
    events = []
    args = SimpleNamespace(root=tmp_path, source_root=tmp_path / 'old')
    provenance = {'source_root': str(args.source_root)}

    def prepare(source, received):
        assert source == args.source_root and received is args
        events.append('reuse')
        return provenance

    def verify(received, evidence, require_banks):
        assert received is args and evidence is provenance and require_banks is True
        events.append('verify')

    def prime(received, evidence):
        assert received is args and evidence is provenance
        events.append('new_vectors')

    @contextmanager
    def server(received):
        events.append('server_start')
        yield
        events.append('server_stop')

    monkeypatch.setattr(pipeline.driver, 'attack', lambda unused: events.append('attack'))
    monkeypatch.setattr(pipeline.driver, 'finalize', lambda unused: events.append('finalize'))
    monkeypatch.setattr(pipeline.driver, 'check', lambda unused: events.append('check'))
    monkeypatch.setattr(pipeline.driver, 'evaluate', lambda unused: events.append('evaluate'))
    monkeypatch.setattr(pipeline.driver, 'running_server', server)
    monkeypatch.setattr(pipeline.driver, 'collect', lambda *args: pytest.fail('Recollection is forbidden'))
    result = {'status': 'completed', 'groups': [1, 2, 6], 'target_purchase_rate': 0.5}
    monkeypatch.setattr(pipeline.driver, 'status', lambda unused: result)
    assert pipeline.run_pipeline(args, prepare_history=prepare, verify_history=verify, prime=prime) == result
    assert events == ['reuse', 'attack', 'finalize', 'verify', 'new_vectors', 'check',
                      'server_start', 'evaluate', 'server_stop', 'verify']


def test_vector_preparation_failure_prevents_remote_shopping(tmp_path, monkeypatch):
    args = SimpleNamespace(root=tmp_path, source_root=tmp_path / 'old')
    monkeypatch.setattr(pipeline.driver, 'attack', lambda unused: None)
    monkeypatch.setattr(pipeline.driver, 'finalize', lambda unused: None)
    monkeypatch.setattr(pipeline.driver, 'evaluate', lambda *unused: pytest.fail('Shopping started with invalid vectors'))
    def broken(*unused):
        raise ValueError('Query image does not match old B')
    with pytest.raises(ValueError, match='old B'):
        pipeline.run_pipeline(args, prepare_history=lambda *unused: {},
                              verify_history=lambda *unused, **kwargs: None, prime=broken)


@pytest.mark.parametrize('same_run', [True, False])
def test_evaluation_resume_uses_frozen_images_and_same_protocol_when_unchanged(tmp_path, monkeypatch, same_run):
    from src.environment.local_webshop import read, write
    events = []
    args = SimpleNamespace(root=tmp_path)
    original = {'status': 'completed', 'original_experiment': True}
    write(tmp_path / 'pipeline_result.json', original)
    write(tmp_path / 'run_settings.json', {'revision': 'current' if same_run else 'previous'})
    monkeypatch.setattr(pipeline.driver, 'load_plan', lambda received: (tmp_path, {'frozen': True}, ['1']))
    monkeypatch.setattr(pipeline.driver, 'execution_settings', lambda *unused: {'revision': 'current'})
    monkeypatch.setattr(pipeline.driver, 'verify_frozen_history_for_evaluation',
                        lambda *unused: events.append('verify_frozen'))
    for name in ('attack', 'collect', 'finalize'):
        monkeypatch.setattr(pipeline.driver, name, lambda *unused: pytest.fail('Historical work repeated'))

    @contextmanager
    def server(received):
        assert received.reuse_frozen_memory is (not same_run) and received.stage == 'evaluate'
        events.append('server_start')
        yield
        events.append('server_stop')

    monkeypatch.setattr(pipeline.driver, 'running_server', server)
    monkeypatch.setattr(pipeline.driver, 'evaluate', lambda unused: events.append('evaluate'))
    result = {'status': 'completed', 'groups': [1], 'executed_primary_tests': 20,
              'target_purchase_rate': 0.5}
    monkeypatch.setattr(pipeline.driver, 'status', lambda unused: result)
    assert pipeline.resume_evaluation(args) == result
    assert events == ([] if same_run else ['verify_frozen']) + ['server_start', 'evaluate', 'server_stop']
    assert read(tmp_path / 'pipeline_result.json') == (result if same_run else original)
    assert read(tmp_path / 'evaluation_resume_result.json') == result


def test_evaluation_resume_rejects_changed_frozen_artifacts_before_model_calls(tmp_path, monkeypatch):
    from src.environment.local_webshop import write
    args = SimpleNamespace(root=tmp_path)
    write(tmp_path / 'run_settings.json', {'previous': True})
    monkeypatch.setattr(pipeline.driver, 'load_plan', lambda unused: (tmp_path, {}, []))
    monkeypatch.setattr(pipeline.driver, 'execution_settings', lambda *unused: {'changed': True})
    def invalid(*unused):
        raise ValueError('Frozen image changed')
    monkeypatch.setattr(pipeline.driver, 'verify_frozen_history_for_evaluation', invalid)
    monkeypatch.setattr(pipeline.driver, 'evaluate', lambda *unused: pytest.fail('Invalid image evaluated'))
    with pytest.raises(ValueError, match='Frozen image'):
        pipeline.resume_evaluation(args)


@pytest.mark.parametrize('method, expected_frequency', [('frequency_momentum', 0.25), ('ssa', None)])
def test_old_b_profile_preserves_comparison_and_selects_only_attack(tmp_path, monkeypatch, method,
                                                                 expected_frequency):
    previous = {'catalog': str(tmp_path / 'catalog'), 'num_steps': 20,
                'memory_prompt_budget': 128000, 'policy_max_tokens': 2048,
                'normal_retries': 2, 'history_max_attempts': 3, 'history_product_mode': 'observed',
                'protocol': pipeline.driver.ATTACKER_PROTOCOL, 'attack_method': 'legacy_pixel',
                'attack_steps': 1000, 'memory_top_k': 1, 'memory_mode': 'full',
                'model': {'name': 'fixture'}, 'template': {'sha256': 'fixture'}, 'epsilon_levels': 16}
    plan = {'webshop_source': str(tmp_path / 'webshop'), 'pair_file': str(tmp_path / 'pairs.json'),
            'seed': 42, 'normal_candidate_limit': 80}
    monkeypatch.setattr(pipeline, 'read', lambda path: plan if path.name == 'manifest.json' else previous)
    monkeypatch.setattr(pipeline.driver, 'load_auxiliary', lambda *unused: ('prompt', previous['template']))
    monkeypatch.setattr(pipeline.driver, 'load_model_config', lambda *unused:
                        SimpleNamespace(public=lambda: previous['model']))
    args = pipeline.profile_arguments(tmp_path / 'source', tmp_path / 'destination', method=method)
    assert args.method == method and args.attack_steps == 500
    assert args.frequency_axis_ratio == expected_frequency
    assert args.group == ['1', '2', '6'] and args.retrieval_encoder == 'clip'
    assert args.reference_overrides is None and args.source_history_mode == 'attacker'
    assert args.include_controls and args.reserve_tests == 0
    assert args.memory_top_k == 1 and args.memory_key_mode == args.memory_query_mode == 'joint'


@pytest.mark.parametrize('options', [
    {'method': 'ssa', 'frequency_axis_ratio': 0.25},
    {'method': 'frequency_momentum', 'ssa_samples': 20},
    {'method': 'ssa', 'ssa_samples': 0},
    {'method': 'ssa', 'ssa_rho': float('nan')},
    {'method': 'ssa', 'attack_steps': 0},
])
def test_old_b_profile_rejects_incompatible_or_invalid_options_before_reading(tmp_path, options):
    with pytest.raises(ValueError):
        pipeline.profile_arguments(tmp_path / 'missing_source', tmp_path / 'destination', **options)
