#!/usr/bin/env python3
"""TurtleBot3 test runner: pytest on one or more folders of tests/.

Usage:
    python3 run_tests.py -<folder> [-<folder> ...] [-v] [-x]
    python3 run_tests.py --list

-<folder>   run all tests in tests/<folder>/
-v          verbose output (passed to pytest)
-x          stop on first failure (passed to pytest)
--list      list the available test folders

Example:
    python3 run_tests.py -lidar_cartographer -gazebo_sim -v
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
    """Split argv into (folders, pytest_flags), ignoring '&' tokens."""
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
    """Run all tests in a folder and return the pytest exit code."""
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
