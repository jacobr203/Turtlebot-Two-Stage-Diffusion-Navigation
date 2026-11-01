"""Pure helpers computing per-trial numbers."""

import csv
import math
import re

from common.geometry import wrap_angle


def percentile(values, q):
    """q-th percentile (0-100) by linear interpolation; None if empty."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def latency_stats(lat_csv):
    """Mean sampling/planning latency, worst tick, p95 scan age."""
    try:
        rows = list(csv.DictReader(open(lat_csv)))
    except OSError:
        return {}
    if not rows:
        return {}
    samp = [float(r['sample_ms']) for r in rows]
    plan = [float(r['plan_ms']) for r in rows]
    out = {'lat_sample_ms': round(sum(samp) / len(samp), 1),
           'lat_plan_ms': round(sum(plan) / len(plan), 1),
           'lat_max_ms': round(max(s + p for s, p in zip(samp, plan)), 1)}
    ages = [float(r['scan_age_ms']) for r in rows if r.get('scan_age_ms')]
    if ages:
        out['scan_age_p95_ms'] = round(percentile(ages, 95), 1)
    return out


_PING_RTT = re.compile(r'time[=<]([\d.]+) ms')


def ping_rtts(text):
    return [float(m) for m in _PING_RTT.findall(text)]


def count_lines(path, needle):
    try:
        return sum(needle in line for line in open(path, errors='replace'))
    except OSError:
        return ''


def relative_pose(origin, pose):
    """pose expressed in the frame of origin (both (x, y, yaw))."""
    ox, oy, oyaw = origin
    dx, dy = pose[0] - ox, pose[1] - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy, wrap_angle(pose[2] - oyaw))


def offset_from_return(remote_s, local_return_s):
    """Remote minus local clock, from the remote time printed at the end of
    a call and the local time it returned."""
    return remote_s - local_return_s
