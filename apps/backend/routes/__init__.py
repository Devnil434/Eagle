"""
Auto-discover and register all routers in this package.
Each module may expose one or more APIRouter instances.
"""
from __future__ import annotations
import importlib
import pkgutil
import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

_discovered_router = APIRouter()


def _autodiscover():
    package_dir = __path__[0]
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name in ("__init__",):
            continue
        module = importlib.import_module(f"{__name__}.{module_name}")
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, APIRouter):
                _discovered_router.include_router(obj)
                logger.debug("Discovered router %s from %s", attr_name, module_name)


_autodiscover()

router = _discovered_router
