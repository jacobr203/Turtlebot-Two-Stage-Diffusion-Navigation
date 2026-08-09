#!/usr/bin/env python3
"""
run_tests.py — TurtleBot3 unit test runner

Usage
-----
Run one folder:
    python3 run_tests.py -lidar_cartographer

Run multiple folders:
    python3 run_tests.py -lidar_cartographer & -another_folder & -third_folder

Flags
-----
-<folder>     Run all tests in ~/unit_tests/<folder>/
-v            Verbose output (passed to pytest)
-x            Stop on first failure (passed to pytest)
--list        List all available test folders without running anything

The & token between folder names is treated as a separator — it is stripped
before arguments are parsed so the command reads naturally.

Examples
--------
    python3 run_tests.py -lidar_cartographer
    python3 run_tests.py -lidar_cartographer & -lidar_cartographer
    python3 run_tests.py -lidar_cartographer -v -x
    python3 run_tests.py --list
"""

import sys
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).parent


def available_folders() -> list[Path]:
    return sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and not p.name.startswith('_') and not p.name.startswith('.')
    )


def parse_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """
    Split argv into (folders, pytest_flags).
    Strips leading '-' from folder names and ignores '&' tokens.
    Passes -v and -x directly to pytest.
    """
    folders: list[str] = []
    flags:   list[str] = []

    pytest_passthrough = {'-v', '-x', '--tb=short', '--tb=long', '--tb=no'}

    for arg in argv:
        if arg == '&':
            continue
        if arg == '--list':
            flags.append('--list')
        elif arg in pytest_passthrough:
            flags.append(arg)
        elif arg.startswith('-'):
            folders.append(arg.lstrip('-'))
        else:
            folders.append(arg)

    return folders, flags


def run_folder(folder_path: Path, pytest_flags: list[str]) -> int:
    """Run all tests in a folder. Returns pytest exit code."""
    print(f'\n{"═" * 60}')
    print(f'  Running tests in: {folder_path.name}/')
    print(f'{"═" * 60}')

    test_files = sorted(folder_path.glob('test_*.py'))
    if not test_files:
        print(f'  WARNING: no test_*.py files found in {folder_path}')
        return 0

    print(f'  Tests found: {[f.name for f in test_files]}')

    env = os.environ.copy()
    env['PYTHONPATH'] = str(BASE) + ':' + env.get('PYTHONPATH', '')

    cmd = [
        sys.executable, '-m', 'pytest',
        str(folder_path),
        '--tb=short',
    ] + pytest_flags

    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    argv = sys.argv[1:]

    if not argv or '--help' in argv or '-h' in argv:
        print(__doc__)
        sys.exit(0)

    folders, flags = parse_args(argv)

    if '--list' in flags:
        print('Available test folders:')
        for f in available_folders():
            tests = list(f.glob('test_*.py'))
            print(f'  -{f.name:<30} ({len(tests)} test file(s))')
        sys.exit(0)

    if not folders:
        print('ERROR: no test folders specified. Use --list to see available folders.')
        sys.exit(1)

    # Validate all folders exist before running any
    missing = [f for f in folders if not (BASE / f).is_dir()]
    if missing:
        print(f'ERROR: folder(s) not found: {missing}')
        print('Use --list to see available folders.')
        sys.exit(1)

    overall_code = 0
    results: list[tuple[str, str]] = []

    for folder_name in folders:
        folder_path = BASE / folder_name
        code = run_folder(folder_path, [f for f in flags if f != '--list'])
        status = 'PASSED' if code == 0 else 'FAILED'
        results.append((folder_name, status))
        if code != 0:
            overall_code = code

    # Summary
    print(f'\n{"═" * 60}')
    print('  Summary')
    print(f'{"═" * 60}')
    for name, status in results:
        icon = '✓' if status == 'PASSED' else '✗'
        print(f'  {icon}  {name:<35} {status}')
    print(f'{"═" * 60}\n')

    sys.exit(overall_code)


if __name__ == '__main__':
    main()
