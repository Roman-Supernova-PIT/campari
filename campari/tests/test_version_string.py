def test_version_is_string():
    from campari import __version__
    assert isinstance(__version__, str)
