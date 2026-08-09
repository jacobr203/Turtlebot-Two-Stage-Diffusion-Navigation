#!/usr/bin/env python3
"""Prove the mixin split changed no method body.

For every class member of the ORIGINAL DiffusionPolicyNode, find it in
the new layout and compare ast.dump() (structure) AND the exact source
segment (bytes). Any difference is a hard failure.
"""
import ast
import sys
from pathlib import Path

ORIG = Path('/tmp/policy_node_PREREFACTOR.py')
DEPLOY = Path('/home/turtlebot/pathfinding_algorithms/capstone/deploy')
NEW_NODE = DEPLOY / 'policy_node.py'
POLICY = DEPLOY / 'policy'


def members_of_class(path, clsname=None):
    src = path.read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if clsname and node.name != clsname:
            continue
        for m in node.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[m.name] = (m, src)
            elif isinstance(m, ast.Assign):
                for t in m.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = (m, src)
    return out


orig = members_of_class(ORIG, 'DiffusionPolicyNode')

new = {}
dupes = []
for p in sorted(POLICY.glob('*.py')) + [NEW_NODE]:
    if p.name == '__init__.py':
        continue
    for k, v in members_of_class(p).items():
        if k in new:
            dupes.append((k, p.name))
        new[k] = v

print(f'original members : {len(orig)}')
print(f'new members      : {len(new)}')

fail = []
if dupes:
    fail.append(f'DUPLICATE members across modules (MRO hazard): {dupes}')

missing = set(orig) - set(new)
extra = set(new) - set(orig)
if missing:
    fail.append(f'MISSING from new layout: {sorted(missing)}')
if extra:
    fail.append(f'UNEXPECTED new members: {sorted(extra)}')

struct_diff, byte_diff = [], []
for name, (onode, osrc) in orig.items():
    if name not in new:
        continue
    nnode, nsrc = new[name]
    if ast.dump(onode) != ast.dump(nnode):
        struct_diff.append(name)
    if ast.get_source_segment(osrc, onode) != \
            ast.get_source_segment(nsrc, nnode):
        byte_diff.append(name)

if struct_diff:
    fail.append(f'AST STRUCTURE CHANGED: {struct_diff}')
if byte_diff:
    fail.append(f'SOURCE BYTES CHANGED: {byte_diff}')

print(f'AST-identical    : {len(orig) - len(struct_diff)}/{len(orig)}')
print(f'byte-identical   : {len(orig) - len(byte_diff)}/{len(orig)}')

if fail:
    print('\n=== FAIL ===')
    for f in fail:
        print(' ', f)
    sys.exit(1)
print('\nPASS — every member present exactly once, '
      'AST- and byte-identical to the original.')
