import pytest

import swarmscope as ss


@pytest.fixture
def sdk():
    s = ss.Swarmscope("memory://", flush_interval=0.01)
    yield s
    s.close()


@pytest.fixture
def sqlite_sdk(tmp_path):
    s = ss.Swarmscope(f"sqlite:///{tmp_path / 't.db'}", flush_interval=0.01)
    yield s
    s.close()
