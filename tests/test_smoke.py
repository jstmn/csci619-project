import pusht619


def test_version() -> None:
    assert isinstance(pusht619.__version__, str)
    assert pusht619.__version__.count(".") == 2
