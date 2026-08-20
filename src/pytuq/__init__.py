#!/usr/bin/env python
"""PyTUQ - Python Toolkit for Uncertainty Quantification."""

from ._version import get_version, get_version_info, get_git_info
from ._tpl_versions import get_all_dependency_versions, get_dependency_versions

# Main version attribute
__version__ = get_version()


def get_version_info():
    """Get comprehensive version information about PyTUQ.
    
    Returns:
        dict: Dictionary containing version information including:
            - version: PyTUQ version string
            - source: How version was determined (importlib_metadata, git, pyproject.toml, fallback)
            - install_method: Installation method (installed, source, unknown)
            - git_commit: Git commit hash (if available)
            - git_tag: Git tag (if available)
    """
    version_info = _get_version_info()
    
    # Add TPL versions to the info
    tpl_versions = get_all_dependency_versions()
    version_info['tpl_versions'] = tpl_versions
    
    return version_info


def _get_version_info():
    """Internal function to get version info without TPL versions."""
    from ._version import get_version_info as _get_base_version_info
    return _get_base_version_info()


def get_tpl_versions():
    """Get versions of third-party libraries used by PyTUQ.
    
    Returns:
        dict: Dictionary containing versions of all dependencies
    """
    return get_all_dependency_versions()

