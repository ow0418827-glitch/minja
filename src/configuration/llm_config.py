"""Private file-based model credentials, compatible with dual-rag's model section."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


from src.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / 'llm.local.json'


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)

    def public(self):
        return {'provider': self.provider, 'model': self.model, 'base_url': self.base_url}

    def client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def load_model_config(path=DEFAULT_CONFIG):
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        raise ValueError('Cannot read model configuration file') from None
    model = raw.get('model') if isinstance(raw, dict) else None
    if not isinstance(model, dict):
        raise ValueError('Configuration requires a model object')
    values = {}
    for name in ('provider', 'model', 'base_url', 'api_key'):
        value = model.get(name)
        if not isinstance(value, str) or not value.strip() or value.strip() == '[REDACTED]':
            raise ValueError('Missing or invalid model.' + name)
        values[name] = value.strip()
    if values['provider'] != 'minimax':
        raise ValueError('This local experiment requires the minimax provider')
    try:
        url = urlsplit(values['base_url'])
        valid = url.scheme in ('https', 'http') and url.hostname and not (
            url.username or url.password or url.query or url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Invalid model.base_url; use a service URL without credentials or query')
    values['base_url'] = values['base_url'].rstrip('/')
    return ModelConfig(**values)
