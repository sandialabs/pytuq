#!/usr/bin/env python
"""Third-party library version detection for PyTUQ."""

import importlib
from typing import Dict, Optional

# Core dependencies
CORE_DEPENDENCIES = ['numpy', 'scipy', 'matplotlib']

# Optional dependencies
OPTIONAL_DEPENDENCIES = ['torch', 'uqinn', 'pyswarms', 'dill']


def _get_package_version(package_name: str) -> Optional[str]:
    """Get version of a package using multiple methods."""
    try:
        # Try importlib.metadata first (modern approach)
        from importlib.metadata import version
        return version(package_name)
    except (ImportError, Exception):
        try:
            # Fallback to package's __version__ attribute
            module = importlib.import_module(package_name)
            return getattr(module, '__version__', None) or getattr(module, 'version', None)
        except (ImportError, AttributeError):
            return None


def get_core_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of core dependencies."""
    versions = {}
    for dep in CORE_DEPENDENCIES:
        versions[dep] = _get_package_version(dep)
    return versions


def get_optional_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of optional dependencies."""
    versions = {}
    for dep in OPTIONAL_DEPENDENCIES:
        versions[dep] = _get_package_version(dep)
    return versions


def get_all_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of all dependencies (core + optional)."""
    versions = {}
    
    # Get core dependencies
    for dep in CORE_DEPENDENCIES:
        versions[dep] = _get_package_version(dep)
    
    # Get optional dependencies
    for dep in OPTIONAL_DEPENDENCIES:
        versions[dep] = _get_package_version(dep)
    
    return versions


def get_dependency_versions() -> Dict[str, Dict[str, Optional[str]]]:
    """Get organized dependency versions."""
    return {
        'core': get_core_dependency_versions(),
        'optional': get_optional_dependency_versions()
    }