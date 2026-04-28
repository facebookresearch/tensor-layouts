# conftest.py — shared fixtures for tests/


def pytest_addoption(parser):
    parser.addoption(
        "--draw",
        action="store_true",
        default=False,
        help="Generate paper figures into tests/figures/",
    )
