import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "xnlp_language" / "generated"


def _generated_hashes() -> dict[str, str]:
    return {
        path.relative_to(GENERATED).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(GENERATED.rglob("*"))
        if path.is_file()
    }


def pytest_sessionstart(session):
    session.config._xnlp_generated_hashes = _generated_hashes()


def pytest_sessionfinish(session, exitstatus):
    before = session.config._xnlp_generated_hashes
    after = _generated_hashes()
    if before != after:
        changed = sorted(
            path for path in set(before) | set(after) if before.get(path) != after.get(path)
        )
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_sep("!", "tests modified xnlp_language/generated/")
            for path in changed:
                reporter.write_line(path)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
