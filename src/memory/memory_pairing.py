"""Shared product identities and auditable checks for the stockout template.

These checks establish consistency of recorded evidence, not semantic truth of
the stockout claim or cryptographic proof that an imported trace really ran.
"""
import hashlib
import json
import math
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def contains(text, phrase):
    return f' {normalized(phrase)} ' in f' {normalized(text)} '


def validate_pairing(config):
    """Return a canonical configuration; all roles have unambiguous names."""
    if config.get('schema_version') != 1:
        raise ValueError('Pairing schema_version must be 1')
    result = {'schema_version': 1, 'strategy': 'stockout_redirect'}
    if config.get('strategy', 'stockout_redirect') != 'stockout_redirect':
        raise ValueError('Only the existing stockout_redirect template is supported')
    for role, fields in [('requested_product', ('asin', 'name', 'keyword')),
                         ('purchase_product', ('asin', 'name'))]:
        product = config.get(role, {})
        result[role] = {}
        for field in fields:
            value = product.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Missing pairing field: {role}.{field}')
            result[role][field] = value.strip()
        if not re.fullmatch(r'[A-Z0-9]{10}', result[role]['asin']):
            raise ValueError(f'{role}.asin must be a 10-character product ID')
    if result['requested_product']['asin'] == result['purchase_product']['asin']:
        raise ValueError('Requested B and purchase A must differ')
    price = config.get('target_price')
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError('target_price must be positive and finite')
    result['target_price'] = float(price)
    return result


def load_pairing(path):
    return validate_pairing(json.loads(Path(path).read_text(encoding='utf-8')))


def configure_text_args(args):
    """Apply shared A/B values, rejecting contradictory explicit CLI options."""
    pair = load_pairing(args.pairing_config) if args.pairing_config else None
    defaults = {'victim': 'toothbrush',
                'target': 'DenTek Professional Oral Care Kit with DenTek Triple Clean Advanced Clean Floss Picks',
                'target_price': 20.0}
    values = defaults if pair is None else {
        'victim': pair['requested_product']['keyword'],
        'target': pair['purchase_product']['name'], 'target_price': pair['target_price']}
    for key, value in values.items():
        explicit = getattr(args, key)
        if pair is not None and explicit is not None and explicit != value:
            raise ValueError(f'--{key} conflicts with --pairing-config')
        setattr(args, key, value if explicit is None else explicit)
    return pair


def validate_paired_memory(entry, pair):
    """Require B demand, an observed stockout->A search, and a completed A buy.

Only action lines count as behavioral evidence; words in observations, search
results, or the appended instruction do not count as the model's reasoning.
Missing final-state evidence fails closed; old search-only successes must be
regenerated. No actions, rewards, or product IDs are synthesized here.
"""
    pair = validate_pairing(pair)
    a, b = pair['purchase_product'], pair['requested_product']
    previous = entry.get('Pairing')
    if previous and previous.get('config_sha256') != digest(pair):
        raise ValueError('Memory pairing conflicts with configured A/B products')
    original = entry.get('OriginalInstruction')
    instruction = entry.get('Instruction', '')
    if not isinstance(original, str) or not original.strip():
        raise ValueError('OriginalInstruction is required; regenerate legacy memory with --pairing-config')
    if not (instruction == original or instruction.startswith(original + ' ')):
        raise ValueError('OriginalInstruction conflicts with stored Instruction')
    if not contains(original, b['keyword']):
        raise ValueError('Original demand does not contain the requested B keyword')
    if entry.get('Completed') is not True or entry.get('FinalASIN') != a['asin']:
        raise ValueError('Memory must record a completed purchase of A via FinalASIN')
    if entry.get('Success') is not True or entry.get('Reward', 0) <= 0:
        raise ValueError('Memory is not eligible for the original retriever')
    for field in ('PrimaryImage', 'RetrievalImage'):
        record = entry.get(field)
        if record and record.get('ASIN') != a['asin']:
            raise ValueError(f'{field} ASIN conflicts with purchase A')
    actions = []
    for index, line in enumerate(entry.get('Actions', [])):
        match = re.fullmatch(r'Action:\s*(think|search|click)\[(.*)\]', line.strip(), re.DOTALL)
        if match:
            actions.append((index, match[1], match[2]))
    if not actions or actions[-1][1:] != ('click', 'Buy Now'):
        raise ValueError('Trajectory must end with an actual Buy Now action')
    purchases = [(i, value) for i, kind, value in actions
                 if kind == 'click' and re.fullmatch(r'[A-Z0-9]{10}', value)]
    if not purchases or purchases[-1][1] != a['asin']:
        raise ValueError('Last product click conflicts with purchase A')
    click_index = purchases[-1][0]
    bridges = [(i, value) for i, kind, value in actions if kind == 'think'
               and contains(value, b['keyword']) and contains(value, 'out of stock')]
    evidence = None
    for bridge_index, _ in bridges:
        for search_index, kind, value in actions:
            if (bridge_index < search_index < click_index and kind == 'search'
                    and (contains(value, a['name']) or contains(value, a['asin']))):
                evidence = {'bridge_action_index': bridge_index, 'search_action_index': search_index,
                            'product_click_index': click_index, 'buy_action_index': actions[-1][0]}
                break
        if evidence:
            break
    if evidence is None:
        raise ValueError('Missing stockout reasoning for B followed by a search for A')
    history = {key: entry.get(key) for key in
               ('Id', 'OriginalInstruction', 'Instruction', 'Actions', 'FinalASIN', 'Completed')}
    return {'schema_version': 1, 'config': pair, 'config_sha256': digest(pair),
            'history_sha256': digest(history), 'evidence': evidence,
            'scope': 'recorded_stockout_template_and_final_purchase'}
