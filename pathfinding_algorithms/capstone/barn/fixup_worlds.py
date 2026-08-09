#!/usr/bin/env python3
"""Produce the two world variants used by this project from the raw BARN set.

  barn/worlds/world_%03d.world       eval variant: physics untouched
                                     (real_time_update_rate 1000 -> RTF <= 1,
                                     required for fair wall-clock inference
                                     latency vs sim-time control comparisons)
  barn/worlds_fast/world_%03d.world  collection variant: real_time_update_rate
                                     0 -> unbounded RTF for headless expert
                                     data collection

Both variants get the gazebo_ros_state world plugin injected (namespace
/gazebo, 30 Hz) so ground-truth poses are available on /gazebo/model_states
and entities can be teleported via /gazebo/set_entity_state while debugging.

Also writes barn/SHA256SUMS (committed) over everything produced, so the
gitignored worlds can be verified as byte-identical after regeneration.

Usage: .venv/bin/python barn/fixup_worlds.py [--raw-dir DIR]
"""

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
RAW_DEFAULT = HERE / 'raw/jackal_helper/worlds/BARN'

STATE_PLUGIN = '''<plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'>
  <ros><namespace>/gazebo</namespace></ros>
  <update_rate>30.0</update_rate>
</plugin>'''


# The shipped BARN gui camera sits ~11 m from the spawn — a 14 cm Burger is
# an invisible speck. Frame the course instead: behind the start, elevated,
# looking up the corridor (+y).
GUI_CAMERA_POSE = '-2.25 -2.0 6.0 0 0.62 1.5708'


def transform(src, dst, unbounded_rtf):
    tree = ET.parse(src)
    world = tree.getroot().find('world')

    if world.find("plugin[@name='gazebo_ros_state']") is None:
        world.insert(0, ET.fromstring(STATE_PLUGIN))

    if unbounded_rtf:
        for phys in world.findall('physics'):
            rate = phys.find('real_time_update_rate')
            if rate is None:
                rate = ET.SubElement(phys, 'real_time_update_rate')
            rate.text = '0'

    cam_pose = world.find('gui/camera/pose')
    if cam_pose is not None:
        cam_pose.text = GUI_CAMERA_POSE

    tree.write(dst, encoding='utf-8', xml_declaration=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, default=RAW_DEFAULT)
    args = ap.parse_args()

    worlds = sorted(args.raw_dir.glob('world_*.world'),
                    key=lambda p: int(p.stem.split('_')[1]))
    if not worlds:
        sys.exit(f'no worlds under {args.raw_dir} — run download_barn.sh first')

    out_eval = HERE / 'worlds'
    out_fast = HERE / 'worlds_fast'
    out_eval.mkdir(exist_ok=True)
    out_fast.mkdir(exist_ok=True)

    produced = []
    for wf in worlds:
        idx = int(wf.stem.split('_')[1])
        for out_dir, fast in ((out_eval, False), (out_fast, True)):
            dst = out_dir / f'world_{idx:03d}.world'
            transform(wf, dst, unbounded_rtf=fast)
            produced.append(dst)

    sums = HERE / 'SHA256SUMS'
    with open(sums, 'w') as f:
        for p in sorted(produced):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            f.write(f'{digest}  {p.relative_to(HERE)}\n')

    print(f'worlds written: {len(produced)} '
          f'({len(worlds)} eval + {len(worlds)} fast)')
    print(f'checksums: {sums} ({len(produced)} entries)')


if __name__ == '__main__':
    main()
