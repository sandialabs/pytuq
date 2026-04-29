#!/usr/bin/env python
"""Version detection and management for PyTUQ."""

import os
import sys
import subprocess
from typing import Dict, Optional, Tuple

# Cache for version information to avoid repeated computations
_version_cache = {}


def _try_importlib_metadata() -> Optional[Dict]:
    """Try to get version info using importlib.metadata (for installed packages)."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            pytuq_version = version('pytuq')
            return {
                'version': pytuq_version,
                'source': 'importlib_metadata',
                'install_method': 'installed'
            }
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _try_git_version() -> Optional[Dict]:
    """Try to get version info from git repository."""
    # Check if we're in a git repository
    # Navigate from src/pytuq/_version.py to project root (should be /projects/pytuq)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    git_dir = os.path.join(project_root, '.git')
    
    if not os.path.exists(git_dir):
        return None

    try:
        # Get git commit hash
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )

        if commit_result.returncode != 0:
            return None

        git_commit = commit_result.stdout.strip()

        # Try to get git tag
        tag_result = subprocess.run(
            ['git', 'describe', '--tags'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )

        git_tag = tag_result.stdout.strip() if tag_result.returncode == 0 else None

        return {
            'version': git_tag or git_commit,
            'git_commit': git_commit,
            'git_tag': git_tag,
            'source': 'git',
            'install_method': 'source'
        }
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return None


def _try_pyproject_toml() -> Optional[Dict]:
    """Try to get version info from pyproject.toml."""
    # Navigate from src/pytuq/_version.py to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    
    if not os.path.exists(pyproject_path):
        return None

    try:
        # Simple parsing for version - avoid external dependencies
        with open(pyproject_path, 'r') as f:
            for line in f:
                if line.strip().startswith('version = '):
                    # Extract version value, handling both single and double quotes
                    version_value = line.split('=')[1].strip()
                    # Remove surrounding quotes
                    version = version_value.strip('"').strip("'")
                    return {
                        'version': version,
                        'source': 'pyproject.toml',
                        'install_method': 'source'
                    }
        return None
    except (IOError, OSError):
        return None


def _get_version_info() -> Dict:
    """Get comprehensive version information."""
    if '_version_info' in _version_cache:
        return _version_cache['_version_info']

    # Try different methods in priority order
    version_info = _try_importlib_metadata()
    
    if version_info is None:
        version_info = _try_git_version()

    if version_info is None:
        version_info = _try_pyproject_toml()

    if version_info is None:
        # Final fallback
        version_info = {
            'version': 'unknown',
            'source': 'fallback',
            'install_method': 'unknown'
        }

    _version_cache['_version_info'] = version_info
    return version_info


def get_version() -> str:
    """Get the PyTUQ version string."""
    version_info = _get_version_info()
    return version_info['version']


def get_version_info() -> Dict:
    """Get comprehensive version information including source and installation method."""
    return _get_version_info()


def get_git_info() -> Dict:
    """Get git-specific version information."""
    version_info = _get_version_info()
    
    git_info = {}
    if 'git_commit' in version_info:
        git_info['commit'] = version_info['git_commit']
    if 'git_tag' in version_info:
        git_info['tag'] = version_info['git_tag']
    
    return git_info