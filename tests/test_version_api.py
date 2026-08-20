#!/usr/bin/env python
"""Test script for PyTUQ version API."""

import sys
import os

def test_version_api():
    """Test the PyTUQ version API."""
    import pytuq
    
    print("=== PyTUQ Version API Test ===")
    
    # Test 1: Basic version attribute
    print(f"1. __version__ attribute: {pytuq.__version__}")
    assert hasattr(pytuq, '__version__'), "Missing __version__ attribute"
    assert isinstance(pytuq.__version__, str), "__version__ should be a string"
    
    # Test 2: get_version() function
    version = pytuq.get_version()
    print(f"2. get_version(): {version}")
    assert version == pytuq.__version__, "get_version() should match __version__"
    
    # Test 3: get_version_info() function
    version_info = pytuq.get_version_info()
    print(f"3. get_version_info(): {version_info}")
    assert isinstance(version_info, dict), "get_version_info() should return a dict"
    assert 'version' in version_info, "Missing 'version' in version_info"
    assert 'source' in version_info, "Missing 'source' in version_info"
    assert 'install_method' in version_info, "Missing 'install_method' in version_info"
    
    # Test 4: get_git_info() function
    git_info = pytuq.get_git_info()
    print(f"4. get_git_info(): {git_info}")
    assert isinstance(git_info, dict), "get_git_info() should return a dict"
    
    # Test 5: get_tpl_versions() function
    tpl_versions = pytuq.get_tpl_versions()
    print(f"5. get_tpl_versions(): {tpl_versions}")
    assert isinstance(tpl_versions, dict), "get_tpl_versions() should return a dict"
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    test_version_api()