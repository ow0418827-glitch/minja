"""Bounded recovery and safe evidence for reasoning-only policy responses."""
import copy
import hashlib
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from src.agent import minja


PRIVATE_REASONING = 'private reasoning body must never become an action or persisted text'


def response(content, *, finish_reason='stop', reasoning=PRIVATE_REASONING,
             completion_tokens=2048):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=reasoning),
            finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=88, completion_tokens=completion_tokens,
            total_tokens=88 + completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2000)))


def fake_client(sequence):
    calls = []

    def create(**request):
        calls.append(copy.deepcopy(request))
        value = sequence[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        return value

    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))), calls


@pytest.fixture
def sleeps(monkeypatch):
    observed = []
    monkeypatch.setattr(minja.time, 'sleep', observed.append)
    return observed


@pytest.mark.parametrize('empty', [None, '', ' \n\t'])
def test_empty_length_grows_budget_and_saves_safe_response_evidence(empty, sleeps):
    client, calls = fake_client([
        response(empty, finish_reason='length'),
        response(empty, finish_reason='length', completion_tokens=4096),
        response('search[shampoo]', completion_tokens=150)])
    trace = {}

    result = minja.llm('Buy shampoo', 'MiniMax-M3', client, 'minimax',
                       trace=trace, max_tokens=2048, max_retries=3)

    assert result == 'search[shampoo]'
    assert [call['max_tokens'] for call in calls] == [2048, 4096, 8192]
    assert len(sleeps) == 2
    assert trace['FinishReason'] == 'stop'
    assert trace['RawResponse'] == trace['ParsedAction'] == 'search[shampoo]'
    attempts = trace['ResponseAttempts']
    assert [item['Attempt'] for item in attempts] == [1, 2, 3]
    assert [item['MaxTokens'] for item in attempts] == [2048, 4096, 8192]
    assert [item['FinishReason'] for item in attempts] == ['length', 'length', 'stop']
    assert attempts[0]['ContentCharacters'] == len(empty or '')
    assert attempts[0]['ReasoningPresent'] is True
    assert attempts[0]['Usage'] == {
        'PromptTokens': 88, 'CompletionTokens': 2048,
        'TotalTokens': 2136, 'ReasoningTokens': 2000}
    assert attempts[0]['ErrorType'] and attempts[1]['ErrorType']
    assert not attempts[2].get('ErrorType')
    assert PRIVATE_REASONING not in json.dumps(trace)
    assert calls[0]['extra_body']['reasoning_split'] is True


def test_empty_stop_retries_same_budget_and_never_executes_reasoning(sleeps):
    client, calls = fake_client([
        response(None, reasoning='click[Buy Now]'),
        response('', reasoning='search[wrong product]'),
        response('think[Check the requested item]')])
    trace = {}
    assert minja.llm('Buy shampoo', 'MiniMax-M3', client, 'minimax',
                     trace=trace, max_tokens=2048, max_retries=3) == 'think[Check the requested item]'
    assert [call['max_tokens'] for call in calls] == [2048, 2048, 2048]
    assert len(trace['ResponseAttempts']) == 3
    assert 'click[Buy Now]' not in json.dumps(trace)
    assert 'search[wrong product]' not in json.dumps(trace)


def test_api_exception_retries_same_budget_and_keeps_failure_metadata(sleeps):
    client, calls = fake_client([ValueError('temporary provider failure'),
                                response('click[< Prev]')])
    trace = {}
    assert minja.llm('task', 'MiniMax-M3', client, 'minimax', trace=trace,
                     max_tokens=2048, max_retries=3) == 'click[< Prev]'
    assert [call['max_tokens'] for call in calls] == [2048, 2048]
    assert trace['ResponseAttempts'][0]['ErrorType'] == 'ValueError'
    assert len(sleeps) == 1


def test_reasoning_only_failures_are_bounded_and_expose_audit_trace(sleeps):
    client, calls = fake_client([response(None, finish_reason='length')] * 3)
    trace = {}
    with pytest.raises(minja.PolicyActionError) as caught:
        minja.llm('task', 'MiniMax-M3', client, 'minimax', trace=trace,
                  max_tokens=2048, max_retries=3)
    assert isinstance(caught.value, RuntimeError)
    assert caught.value.recoverable is True
    assert caught.value.trace == trace
    assert len(calls) == len(trace['ResponseAttempts']) == 3
    assert [call['max_tokens'] for call in calls] == [2048, 4096, 8192]
    assert len(sleeps) == 2
    assert PRIVATE_REASONING not in str(caught.value)
    assert PRIVATE_REASONING not in json.dumps(trace)


def test_missing_choices_is_a_bounded_audited_empty_response(sleeps):
    client, calls = fake_client([SimpleNamespace(choices=[], usage=None)] * 2)
    trace = {}
    with pytest.raises(minja.PolicyActionError) as caught:
        minja.llm('task', 'MiniMax-M3', client, 'minimax', trace=trace,
                  max_tokens=2048, max_retries=2)
    assert caught.value.recoverable is True
    assert caught.value.trace == trace
    assert len(calls) == len(trace['ResponseAttempts']) == 2
    assert [call['max_tokens'] for call in calls] == [2048, 2048]
    assert all(item['FinishReason'] is None for item in trace['ResponseAttempts'])
    assert all(item['ErrorType'] for item in trace['ResponseAttempts'])
    assert len(sleeps) == 1


def test_last_failed_attempt_does_not_sleep(sleeps):
    client, calls = fake_client([response(None, finish_reason='length')])
    with pytest.raises(minja.PolicyActionError):
        minja.llm('task', 'MiniMax-M3', client, 'minimax',
                  max_tokens=2048, max_retries=1)
    assert len(calls) == 1
    assert sleeps == []


def test_base_budget_above_recovery_ceiling_is_never_lowered(sleeps):
    client, calls = fake_client([response(None, finish_reason='length'),
                                response('search[shampoo]')])
    assert minja.llm('task', 'MiniMax-M3', client, 'minimax',
                     max_tokens=16384, max_retries=2) == 'search[shampoo]'
    assert [call['max_tokens'] for call in calls] == [16384, 16384]


def test_retry_keeps_identical_query_and_recalled_image_payloads(tmp_path, sleeps):
    query = tmp_path / 'query.png'
    recalled = tmp_path / 'recalled.png'
    Image.new('RGB', (8, 7), (20, 120, 240)).save(query)
    Image.new('RGB', (8, 7), (220, 20, 80)).save(recalled)
    assets = [{'Id': 'historical_memory', 'Role': 'recalled_memory',
               'ASIN': 'PRODUCT001', 'LocalPath': str(recalled),
               'SHA256': hashlib.sha256(recalled.read_bytes()).hexdigest()}]
    client, calls = fake_client([response(None, finish_reason='length'),
                                response('search[shampoo]')])
    trace = {}

    assert minja.llm('Memory Id: historical_memory\nBuy shampoo', 'MiniMax-M3',
                     client, 'minimax', query_image=query, memory_images=assets,
                     trace=trace, max_tokens=2048, max_retries=2) == 'search[shampoo]'

    assert calls[0]['messages'] == calls[1]['messages']
    blocks = calls[0]['messages'][1]['content']
    assert len([block for block in blocks if block['type'] == 'image_url']) == 2
    assert [item['Role'] for item in trace['InputImages']] == ['query', 'recalled_memory']
    assert PRIVATE_REASONING not in json.dumps(trace)


@pytest.mark.parametrize('failure', ['empty_action', 'timeout'])
def test_policy_failure_keeps_partial_episode_without_creating_a_purchase(tmp_path, monkeypatch, sleeps, failure):
    monkeypatch.setattr(minja, 'webshop_text', lambda **kwargs: ('Shopping page', {'instruction': 'Buy shampoo'}))
    value = response(None, finish_reason='length') if failure == 'empty_action' else TimeoutError('network timeout')
    client, _ = fake_client([value])
    expected = minja.PolicyActionError if failure == 'empty_action' else TimeoutError
    with pytest.raises(expected) as caught:
        minja.webshop_run_rap(
            'partial_history', '', '', 'Examples', minja.webshopEnv(), client,
            'MiniMax-M3', 'minimax', None, [], {}, 1, honest_reward=True,
            memory_image_input=True, memory_path=tmp_path / 'memory.json',
            policy_max_tokens=2048, policy_max_retries=1, to_print=False)
    partial = caught.value.partial_memory
    assert partial['Id'] == 'partial_history'
    assert partial['Instruction'] == 'Buy shampoo'
    assert not partial['Completed'] and not partial['Success']
    assert partial['FinalASIN'] is None and partial['EnvironmentReward'] is None
    assert partial['TerminationReason'] == 'policy_call_error'
    assert partial['PolicyTrace'][0]['Executed'] is False
    assert partial['PolicyTrace'][0]['ResponseAttempts'][0]['ErrorType']
    assert PRIVATE_REASONING not in json.dumps(partial)
