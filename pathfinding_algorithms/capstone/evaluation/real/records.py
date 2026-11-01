"""Reading and amending the trials CSV."""

import csv
import subprocess

from .config import WS

def git_state():
    """(short sha, dirty) of the code under test (tracked files only)."""
    def git(*a):
        return subprocess.run(['git', '-C', str(WS), *a], capture_output=True,
                              text=True).stdout.strip()
    sha = git('rev-parse', '--short', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no', '--', str(WS))
    return sha, dirty


def add_late_collisions(path, n):
    """Add contacts the operator could not key in to the last row, also in
    their own column because they have no time or place on the path."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    if 'collisions_late' not in header:
        header.append('collisions_late')
    rows = read_rows(path)
    rows[-1]['collisions_late'] = n
    rows[-1]['collisions'] = int(rows[-1].get('collisions') or 0) + n
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def set_last_field(path, key, value):
    """Set one field of the row just appended (adding the column if new)."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    if key not in header:
        header.append(key)
    rows = read_rows(path)
    rows[-1][key] = value
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def set_last_notes(path, notes):
    """Fill in the notes of the row just appended."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    rows = read_rows(path)
    rows[-1]['notes'] = notes
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, restval='')
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    try:
        return list(csv.DictReader(open(path)))
    except OSError:
        return []


def append_row(path, fields, row):
    """Append one row under the file's own header, first rewriting the file
    if the code has gained fields, so old and new rows stay aligned."""
    header = None
    if path.exists():
        with open(path, newline='') as f:
            header = next(csv.reader(f), None)
    if not header:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerow(row)
        return
    union = header + [k for k in fields if k not in header]
    if union != header:
        old = read_rows(path)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=union, restval='')
            w.writeheader()
            w.writerows(old)
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=union, restval='',
                       extrasaction='ignore').writerow(row)
