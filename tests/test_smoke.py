from hiring_cv_bias.config import DATA_DIR


def test_package_imports() -> None:
    assert DATA_DIR.endswith("/data/")
