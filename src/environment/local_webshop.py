"""Isolated small-catalog adapter; leaves the sibling WebShop source unchanged."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def configure_java():
    if os.name == 'nt' and not os.environ.get('JAVA_HOME'):
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\JavaSoft\JDK') as key:
            version = winreg.QueryValueEx(key, 'CurrentVersion')[0]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\JavaSoft\\JDK\\' + version) as key:
            os.environ['JAVA_HOME'] = winreg.QueryValueEx(key, 'JavaHome')[0]
    # Pyserini eagerly initializes an unused API encoder. No model calls here.
    os.environ.setdefault('OPENAI_API_KEY', 'not-used')


def load_catalog(root):
    root = Path(root).resolve()
    config = read(root / 'local_config.json')
    source = Path(config['webshop_source']).resolve()
    sys.path.insert(0, str(source))
    configure_java()
    os.environ['WEBSHOP_DATA_MODE'] = 'full'
    from web_agent_site import utils
    utils.DATA_MODE = 'local_pair'
    utils.DEFAULT_FILE_PATH = str(root / 'data/items.json')
    utils.DEFAULT_ATTR_PATH = str(root / 'data/attributes.json')
    utils.HUMAN_ATTR_PATH = str(root / 'data/human.json')
    utils.DEBUG_PROD_SIZE = None
    utils.SMOKE_TARGET_ASIN = None
    from web_agent_site.engine import engine
    engine.DATA_MODE = utils.DATA_MODE
    engine.DEFAULT_ATTR_PATH = utils.DEFAULT_ATTR_PATH
    engine.HUMAN_ATTR_PATH = utils.HUMAN_ATTR_PATH
    random.seed(config['catalog_seed'])
    products, by_id, prices, attributes = engine.load_products(utils.DEFAULT_FILE_PATH)
    from web_agent_site.engine.goal import get_goals
    goals = get_goals(products, prices)
    random.Random(233).shuffle(goals)
    return config, products, by_id, prices, attributes, goals


def make_site(root):
    root = Path(root).resolve()
    config, products, by_id, prices, attributes, goals = load_catalog(root)
    expected = read(root / 'tasks.json')
    if fingerprint(goals) != expected['goals_sha256']:
        raise ValueError('Goals changed; rebuild the local task bundle before running')
    from pyserini.search.lucene import LuceneSearcher
    from web_agent_site import app as site
    from flask import jsonify, send_from_directory, request
    site.all_products, site.product_item_dict = products, by_id
    site.product_prices, site.attribute_to_asins = prices, attributes
    site.goals, site.weights = goals, [g['weight'] for g in goals]
    site.search_engine = LuceneSearcher(str(root / 'index'))
    site.SHOW_ATTRS_TAB = True

    @site.app.before_request
    def reset_episode():
        if request.endpoint == 'index' and request.method == 'GET':
            # Only our fixed goal sessions are accepted by the driver. A reset
            # recreates the same real goal, removing previous purchase state.
            sid = (request.view_args or {}).get('session_id', '')
            if sid.startswith('fixed_'):
                site.user_sessions.pop(sid, None)

    @site.app.route('/local-pair/health')
    def health():
        return jsonify({'bundle_id': expected['bundle_id'], 'products': len(products),
                        'goals_sha256': expected['goals_sha256'],
                        'a': config['a'], 'b': config['b']})

    @site.app.route('/local-pair/assets/<filename>')
    def asset(filename):
        return send_from_directory(root / 'assets', filename)

    return site, expected


def serve(root):
    site, _ = make_site(root)
    config = read(Path(root) / 'local_config.json')
    site.app.run(host='127.0.0.1', port=config['port'], threaded=False, use_reloader=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    serve(parser.parse_args().root)
