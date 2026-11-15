"""Find checkout data without assuming the installed package owns user files."""

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("COURSE_QA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    for candidate in (checkout, Path.cwd(), Path.cwd() / "course-notes-qa-assistant"):
        if (candidate / "configs" / "app.yaml").is_file():
            return candidate.resolve()
    return Path.cwd()
