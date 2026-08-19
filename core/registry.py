"""
Plugin registry.

This is the single mechanism that makes the agent expandable. Every
capability (data source, feature extractor, model, action handler) is a
class registered here under a name. The Agent never imports concrete
implementations directly - it asks the registry for whatever name is
listed in config/agent_config.yaml.

To add a new capability later:
    1. Write a class implementing the relevant base interface
       (see data/base.py, features/base.py, models/base.py, actions/base.py)
    2. Decorate it with @register("<category>", "<name>")
    3. Add "<name>" to config/agent_config.yaml
That's it - no changes anywhere else in the codebase.
"""

from collections import defaultdict

_REGISTRY = defaultdict(dict)


def register(category: str, name: str):
    """Class decorator: register a plugin implementation under a category/name."""

    def decorator(cls):
        if name in _REGISTRY[category]:
            raise ValueError(f"Duplicate registration: {category}/{name}")
        _REGISTRY[category][name] = cls
        return cls

    return decorator


def get(category: str, name: str):
    """Instantiate and return the registered class for category/name."""
    try:
        cls = _REGISTRY[category][name]
    except KeyError:
        available = list(_REGISTRY[category].keys())
        raise KeyError(
            f"No plugin registered as {category}/{name}. "
            f"Available for '{category}': {available}"
        )
    return cls


def available(category: str):
    """List registered plugin names for a category (useful for debugging/demo)."""
    return list(_REGISTRY[category].keys())


def all_registered():
    """Full map of category -> [names], for printing at startup."""
    return {cat: list(d.keys()) for cat, d in _REGISTRY.items()}
