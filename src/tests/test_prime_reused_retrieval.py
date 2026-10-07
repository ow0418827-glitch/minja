import copy
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.environment.local_webshop import read, write
from src.experiments import prime_reused_retrieval as module
from src.memory.multimodal_memory import apply_controlled_image_substitution
from src.vision.visual_retrieval import canonical_digest, sha256


class FakeEncoder:
    snapshot = {'model': 'test-512-dimensional', 'fusion': 'full-text-and-image'}

    def __init__(self):
        self.closed = False
        self.calls = 0

    def encode_joint(self, text, path):
        self.calls += 1
        raw = text.encode('utf-8') + Path(path).read_bytes()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')
        return np.random.default_rng(seed).normal(size=512).astype(np.float32)

    def close(self):
        self.closed = True


@pytest.fixture
def reuse(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'frequency'
    group_id = '01_camera'
    source_folder, folder = [root / 'groups' / group_id for root in (source, destination)]
    for location in (source_folder, folder):
        (location / 'images').mkdir(parents=True)
        for name, value in [('a', 100), ('query', 200), ('pixel', 105), ('frequency', 110)]:
            Image.new('RGB', (8, 8), (value, 0, 0)).save(location / 'images' / (name + '.png'))
        for index in range(14):
            Image.new('RGB', (8, 8), (140 + index, 0, 0)).save(location / 'images' / f'normal{index}.png')
    def entry(identifier, asin, image):
        primary = {'ASIN': asin, 'LocalPath': 'images/' + image.name, 'SHA256': sha256(image)}
        return {'Id': identifier, 'Instruction': 'Purchase ' + asin,
                'Actions': ['search[' + asin + ']', 'click[Buy Now]'],
                'Success': True, 'Reward': 1.0, 'EnvironmentReward': 1.0,
                'Completed': True, 'FinalASIN': asin,
                'PrimaryImage': primary, 'RetrievalImage': copy.deepcopy(primary)}
    clean = [entry(f'normal-{index}', f'N{index:09d}', source_folder / 'images' / f'normal{index}.png')
             for index in range(14)]
    clean.append(entry('designated-a', 'A000000001', source_folder / 'images/a.png'))
    for location in (source_folder, folder):
        write(location / 'clean_memory.json', clean)
    old = copy.deepcopy(clean)
    old[-1] = apply_controlled_image_substitution(
        old[-1], source_folder / 'images/pixel.png', source_folder / 'clean_memory.json',
        source_asin='A000000001', protocol=module.driver.ATTACKER_PROTOCOL)
    write(source_folder / 'memory.json', old)
    attacked = copy.deepcopy(clean)
    attacked[-1] = apply_controlled_image_substitution(
        attacked[-1], folder / 'images/frequency.png', folder / 'clean_memory.json',
        source_asin='A000000001', protocol=module.driver.ATTACKER_PROTOCOL)
    write(folder / 'memory.json', attacked)
    write(folder / 'memory_ready.json', {
        'source_memory_id': 'designated-a', 'purchased_asin': 'A000000001',
        'depicted_asin': 'A000000001', 'reference_asin': 'B000000001',
        'source_image_sha256': clean[-1]['PrimaryImage']['SHA256'],
        'attack_sha256': attacked[-1]['RetrievalImage']['SHA256'],
        'protocol': module.driver.ATTACKER_PROTOCOL,
        'bank_sha256': {'clean': sha256(folder / 'clean_memory.json'),
                        'adversarial': sha256(folder / 'memory.json')},
    })
    source_group = {'id': group_id, 'a': 'A000000001', 'b': 'B000000001',
                    'query_image': str(source_folder / 'images/query.png')}
    group = {**source_group, 'query_image': str(folder / 'images/query.png')}
    write(source / 'manifest.json', {'groups': [source_group]})
    # An old cache sentinel proves that it is neither read as an index nor edited.
    write(source_folder / 'retrieval_index_cache/sentinel.json', {'vector': ['invalid-old-cache']})
    provenance = {'source_root': str(source), 'source_files_sha256': {
        path.relative_to(source).as_posix(): sha256(path) for path in source.rglob('*') if path.is_file()}}
    tasks = [{'task_id': f'test-{index}', 'instruction': f'Buy the product in this image wording {index}',
              'query_image_sha256': sha256(folder / 'images/query.png')} for index in range(20)]
    args = SimpleNamespace(root=destination, reserve_tests=0, query_specificity='vague',
                           memory_key_mode='joint', memory_query_mode='joint', index_text='full',
                           retrieval_encoder='clip', optimization_config=None, device='cpu')
    encoder = FakeEncoder()
    monkeypatch.setattr(module, 'make_joint_encoder', lambda **kwargs: encoder)
    monkeypatch.setattr(module.driver, 'load_plan', lambda unused: (destination, {}, [group]))
    monkeypatch.setattr(module.driver, 'resolve_history_binding', lambda *args, **kwargs: group)
    monkeypatch.setattr(module.driver, 'evaluation_location', lambda *args, **kwargs: (folder / 'evaluation', {}))
    monkeypatch.setattr(module.driver, 'evaluation_trials', lambda *args, **kwargs: (tasks, 'frozen-task-digest'))
    monkeypatch.setattr(module.driver, 'runtime', lambda *args, **kwargs: pytest.fail('No remote policy runtime is allowed'))
    return SimpleNamespace(source=source, root=destination, folder=folder, args=args,
                           provenance=provenance, tasks=tasks, group=group, encoder=encoder)


def test_twenty_queries_and_fifteen_vectors_per_condition_preserve_source(reuse):
    before = {path.relative_to(reuse.source).as_posix(): path.read_bytes()
              for path in reuse.source.rglob('*') if path.is_file()}
    report = module.prime_retrieval(reuse.args, reuse.provenance)
    group = report['groups'][0]
    assert report['remote_model_calls'] == 0 and report['source_files_unchanged']
    assert not report['old_retrieval_cache_reused']
    assert group['unchanged_normal_memories'] == 14
    assert group['bank_vectors_per_condition'] == {'clean': 15, 'adversarial': 15}
    assert len(group['rankings']['clean']) == len(group['rankings']['adversarial']) == 20
    assert group['source_vector_changed_from_clean'] and group['source_vector_changed_from_old_pixel']
    queries = read(reuse.root / 'query_vector_preparation.json')['groups'][0]['query_vectors']
    assert len(queries) == 20 and all(len(row['vector']) == 512 for row in queries)
    assert all(row['condition_vector_sha256']['clean'] == row['condition_vector_sha256']['adversarial']
               and row['vector_sha256'] == canonical_digest(row['vector']) for row in queries)
    assert all(row['query_text_sha256'] == hashlib.sha256(row['instruction'].encode('utf-8')).hexdigest()
               for row in queries)
    caches = list((reuse.folder / 'retrieval_index_cache').glob('*.json'))
    assert len(caches) == 2 and all(len(read(path)['vectors']) == 15 for path in caches)
    assert all(path.read_bytes() == before[path.relative_to(reuse.source).as_posix()]
               for path in reuse.source.rglob('*') if path.is_file())
    assert len(before) == sum(path.is_file() for path in reuse.source.rglob('*'))
    assert reuse.encoder.closed


def test_wrong_task_query_digest_rejected_and_encoder_closed(reuse):
    reuse.tasks[3]['query_image_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='twenty demands'):
        module.prime_retrieval(reuse.args, reuse.provenance)
    assert reuse.encoder.closed


def test_replacement_b_image_rejected(reuse):
    reuse.group['query_image'] = str(reuse.folder / 'images/normal0.png')
    with pytest.raises(ValueError, match='frozen old B'):
        module.prime_retrieval(reuse.args, reuse.provenance)


def test_changed_original_source_rejected_before_encoder_load(reuse):
    write(reuse.source / 'manifest.json', {'groups': []})
    with pytest.raises(ValueError, match='Original source file changed'):
        module.prime_retrieval(reuse.args, reuse.provenance)
    assert reuse.encoder.calls == 0


def test_normal_history_change_in_both_conditions_rejected(reuse):
    for name in ('clean_memory.json', 'memory.json'):
        bank = read(reuse.folder / name)
        bank[0]['Actions'].append('invented new history')
        write(reuse.folder / name, bank)
    ready = read(reuse.folder / 'memory_ready.json')
    ready['bank_sha256'] = {'clean': sha256(reuse.folder / 'clean_memory.json'),
                           'adversarial': sha256(reuse.folder / 'memory.json')}
    write(reuse.folder / 'memory_ready.json', ready)
    with pytest.raises(ValueError, match='normal memory'):
        module.prime_retrieval(reuse.args, reuse.provenance)


def test_same_frequency_and_old_pixel_vector_is_reported_without_cherry_picking(reuse):
    old_bank = read(reuse.source / 'groups/01_camera/memory.json')
    source_image = Path(old_bank[-1]['RetrievalImage']['LocalPath'])
    destination = reuse.folder / 'images/frequency.png'
    shutil.copyfile(source_image, destination)
    clean = read(reuse.folder / 'clean_memory.json')
    bank = copy.deepcopy(clean)
    bank[-1] = apply_controlled_image_substitution(
        bank[-1], destination, reuse.folder / 'clean_memory.json',
        source_asin='A000000001', protocol=module.driver.ATTACKER_PROTOCOL)
    write(reuse.folder / 'memory.json', bank)
    ready = read(reuse.folder / 'memory_ready.json')
    ready['bank_sha256']['adversarial'] = sha256(reuse.folder / 'memory.json')
    ready['attack_sha256'] = sha256(destination)
    write(reuse.folder / 'memory_ready.json', ready)
    report = module.prime_retrieval(reuse.args, reuse.provenance)
    assert not report['groups'][0]['source_vector_changed_from_old_pixel']
    assert report['groups'][0]['source_vector_changed_from_clean']


def test_normal_image_change_in_both_conditions_rejected(reuse):
    image_path = reuse.folder / 'images/normal0.png'
    Image.new('RGB', (8, 8), (1, 2, 3)).save(image_path)
    for name in ('clean_memory.json', 'memory.json'):
        bank = read(reuse.folder / name)
        for field in ('PrimaryImage', 'RetrievalImage'):
            bank[0][field]['SHA256'] = sha256(image_path)
        write(reuse.folder / name, bank)
    ready = read(reuse.folder / 'memory_ready.json')
    ready['bank_sha256'] = {'clean': sha256(reuse.folder / 'clean_memory.json'),
                           'adversarial': sha256(reuse.folder / 'memory.json')}
    write(reuse.folder / 'memory_ready.json', ready)
    with pytest.raises(ValueError, match='normal memory'):
        module.prime_retrieval(reuse.args, reuse.provenance)
