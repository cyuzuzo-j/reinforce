from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

from polymarket_gym.features.base import FeatureTransform, apply_features

_REGISTRY: dict[str, FeatureTransform] = {}


def register(feature: FeatureTransform) -> FeatureTransform:
    """Decorator: register a feature instance (or class with zero-arg ctor) by name.

    Duplicate names raise — feature files must not silently shadow each other.
    """
    inst = feature() if isinstance(feature, type) else feature
    name = getattr(inst, "name", None)
    if not name:
        raise ValueError(f"feature {inst!r} missing non-empty `name` attribute")
    if name in _REGISTRY:
        raise ValueError(f"feature {name!r} already registered")
    _REGISTRY[name] = inst
    return inst


def get(name: str) -> FeatureTransform:
    discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    discover()
    return sorted(_REGISTRY)


def build(names: list[str]) -> list[FeatureTransform]:
    """Resolve a list of feature names into a pipeline.

    The returned list is the order in which features will be applied, so
    callers can A/B test by varying just this list — no other state moves.
    """
    return [get(n) for n in names]


_discovered = False


def discover() -> None:
    """Import every sibling module so their ``@register`` decorators fire.

    Isolation contract: importing a feature module must not cause side effects
    beyond its own registration. Failures in one module are surfaced, not
    swallowed, so a broken experiment can't silently corrupt the registry.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    import polymarket_gym.features as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_") or mod.name == "base":
            continue
        importlib.import_module(f"{pkg.__name__}.{mod.name}")


__all__ = [
    "FeatureTransform",
    "apply_features",
    "register",
    "get",
    "available",
    "build",
    "discover",
]
