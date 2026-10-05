"""Experiment groups and text-baseline settings (no model loading)."""
EXECUTION_VERSION = 4
IMAGE_CONDITIONS = ('no_memory', 'clean', 'adversarial')


def experiment_conditions(args):
    if getattr(args, 'text_baseline_only', False):
        return ('text',)
    if getattr(args, 'include_text_baseline', False):
        return ('no_memory', 'text', 'clean', 'adversarial')
    return IMAGE_CONDITIONS


def text_top_k(args):
    value = getattr(args, 'text_top_k', 3)
    if not isinstance(value, int) or value < 1:
        raise ValueError('text-top-k must be positive')
    return value
