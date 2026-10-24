import json
import py_compile
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}
HEAVY_MODULES = ("torch", "gymnasium", "stable_baselines3", "encoders.factory", "encoders.cnn")

PROBE = """
import json, sys
import config
print(json.dumps([m for m in sys.modules if m in sys.argv[1].split(",")]))
"""


def py_files(root):
    for path in sorted(root.rglob("*.py")):
        if not SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
            yield path


def check_syntax(root):
    failures = []
    for path in py_files(root):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            error = exc.exc_value
            detail = error.msg if isinstance(error, SyntaxError) else exc.msg.strip()
            line = f":{error.lineno}" if isinstance(error, SyntaxError) else ""
            failures.append(f"{path.relative_to(root)}{line}: {detail}")
    return failures


def check_lazy_imports(root):
    probe = subprocess.run(
        [sys.executable, "-c", PROBE, ",".join(HEAVY_MODULES)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return [f"import config failed: {probe.stderr.strip().splitlines()[-1]}"]
    leaked = json.loads(probe.stdout.strip().splitlines()[-1])
    return [f"import config pulled in {name}" for name in leaked]


CHECKS = (("syntax", check_syntax), ("lazy imports", check_lazy_imports))


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    failed = 0
    for name, check in CHECKS:
        failures = check(root)
        print(f"{'FAIL' if failures else 'ok  '} {name}")
        for failure in failures:
            print(f"     {failure}")
        failed += bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
