"""Basic tests for spectraldesign package."""

import spectraldesign


def test_import():
    """Test that the package can be imported."""
    assert spectraldesign is not None


def test_version():
    """Test that the package has a version."""
    assert hasattr(spectraldesign, "__version__")
    assert isinstance(spectraldesign.__version__, str)
