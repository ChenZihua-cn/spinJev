#!/usr/bin/env python3
"""Draw the per-decision-tick trace of one episode: heights, grip, forces, torque, drift.

The sweep logs every 50 Hz decision tick to data/demo/traces.jsonl. This script turns one
(source, trial) pair into a six-panel diagnostic and prints the event timeline, so a failure can be
read as a mechanism ("the block crossed below hold_z at t=5.12 s while the grip was still stalled at
0.286 rad") instead of a label.

Run:
    .venv/bin/python scripts/plot_trace.py --source single --trial 2
    .venv/bin/python scripts/plot_trace.py --source avg20 --trial 2
    .venv/bin/python scripts/plot_trace.py --source truth --trial 8 --out data/demo/diagnostics/table_hit.png
    .venv/bin/python scripts/plot_trace.py --source single --trial 2 --csv data/demo/diagnostics/single_t02.csv
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinjev.bus.schema import TAU_LIMIT  # noqa: E402
from spinjev.spine.guard import TABLE_FORCE_LIMIT  # noqa: E402
from spinjev.spine.primitives import CLOSE_Q, OPEN_Q  # noqa: E402

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_INDEX = 2
SOURCE_CN = {
    "truth": "真值位姿（仅诊断）",
    "single": "单次检测",
    "avg20": "悬停平均 20 帧",
    "avg60": "悬停平均 60 帧",
}
BG = (14, 16, 20)
PANEL = (19, 22, 27)
GRID = (38, 42, 50)
AXIS = (86, 92, 102)
TEXT = (228, 231, 237)
MUTED = (145, 151, 161)
BLUE = (84, 172, 232)
GREEN = (92, 204, 124)
RED = (235, 92, 82)
ORANGE = (240, 150, 80)
YELLOW = (230, 200, 90)
PURPLE = (178, 140, 235)
CYAN = (100, 210, 210)


def font(size):
    return ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX)


def dashed(draw, p0, p1, fill, dash=6, gap=4, width=1):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    n = int(length // (dash + gap)) + 1
    for i in range(n):
        a = i * (dash + gap) / length
        b = min(1.0, a + dash / length)
        draw.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                   x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=fill, width=width)


def text_r(draw, xy, s, f, fill):
    draw.text(xy, s, font=f, fill=fill, anchor="ra")


def load_episode(args):
    rows = []
    with open(args.trace) as fh:
        for line in fh:
            if f'"source":"{args.source}"' not in line:
                continue
            r = json.loads(line)
            if r["trial"] == args.trial - 1:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no trace rows for {args.source} trial {args.trial}")
    meta = json.load(open(args.meta))
    ep = next((e for e in meta["episodes"]
               if e["source"] == args.source and e["trial"] == args.trial - 1), None)
    if ep is None:
        raise SystemExit("episode missing from trace_meta.json")
    return rows, ep, meta


def series(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def vec(rows, key, idx):
    return np.array([r[key][idx] for r in rows], dtype=float)


def stats(rows, ep, meta):
    t = series(rows, "t")
    z = vec(rows, "block", 2)
    eez = vec(rows, "ee", 2)
    grip = series(rows, "grip")
    gtgt = np.array([r["q_target"][5] for r in rows])
    force = series(rows, "table_force")
    tau = np.array([max(abs(v) for v in r["tau"][:5]) for r in rows])
    xy0 = np.array(rows[0]["block"][:2])
    drift = np.array([np.linalg.norm(np.array(r["block"][:2]) - xy0) * 1000 for r in rows])
    vz = vec(rows, "block_vel", 2)
    hold_z = float(meta["protocol"]["hold_z"])
    rest_z = float(meta["protocol"]["block_rest_z"])

    # Cut the slip window at the start of the RELEASE primitive, not at the sticky "released"
    # event: the jaws take ~1.8 s to open, and the block hitting the table on purpose is not slip.
    release_start = next((e["t"] for e in ep["events"]
                          if e["event"] == "primitive_end" and e.get("next") == "释放"), None)
    release_t = release_start if release_start is not None else t[-1]
    before_release = t <= release_t
    imax = int(np.argmax(z))
    after = z[imax:][before_release[imax:]]
    slip = float((z[imax] - after.min()) * 1000) if len(after) else 0.0
    close_ev = next((e for e in ep["events"]
                     if e["event"] == "primitive_end" and e["prim"] == "闭合"), None)
    grip_close = None
    if close_ev is not None:
        near = min(rows, key=lambda r: abs(r["tick"] - close_ev["tick"]))
        grip_close = near["grip"]
    return {
        "t": t, "z": z, "eez": eez, "grip": grip, "gtgt": gtgt, "force": force,
        "tau": tau, "drift": drift, "vz": vz, "hold_z": hold_z, "rest_z": rest_z,
        "max_z": float(z.max()), "slip_mm": slip, "grip_close": grip_close,
        "max_force": float(force.max()), "max_tau": float(tau.max()),
        "final_drift": float(drift[-1]), "max_held": max(r["held_ticks"] for r in rows),
        "release_t": release_t if release_t != t[-1] else None,
    }


def phase_segments(ep, tmax):
    """[(t0, t1, name)] from the primitive_end events; the tail is the phase that was active
    when the episode ended, not the word "end"."""
    events = [e for e in ep["events"] if e["event"] == "primitive_end"]
    out = []
    t0 = 0.0
    for e in events:
        out.append((t0, e["t"], e["prim"]))
        t0 = e["t"]
    if any(e["event"] == "wait_start" for e in ep["events"]):
        tail = "等待稳定"
    elif events:
        tail = events[-1].get("next", "结束")
    else:
        tail = "结束"
    if tmax > t0:
        out.append((t0, tmax, tail))
    return out


def draw_phase_bar(draw, x0, x1, y, h, tmax, segments):
    cols = [(44, 52, 64), (54, 62, 74), (38, 48, 60), (48, 58, 72)]
    for i, (t0, t1, name) in enumerate(segments):
        a = x0 + (x1 - x0) * (t0 / tmax)
        b = x0 + (x1 - x0) * (t1 / tmax)
        draw.rectangle((a, y, max(a + 1, b), y + h), fill=cols[i % len(cols)])
        if b - a > 30:
            draw.text(((a + b) / 2, y + h / 2), name, font=font(12), fill=MUTED, anchor="mm")


def panel(draw, box, title, t, lines, hlines=(), yfmt="{:.3f}", ylabel="", vlines=(),
          red_vlines=(), notes=()):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill=PANEL)
    draw.text((x0 + 8, y0 + 6), title, font=font(17), fill=TEXT)
    px0, px1 = x0 + 62, x1 - 12
    py0, py1 = y0 + 34, y1 - 46
    vals = np.concatenate([np.asarray(v, dtype=float) for _, _, v, _ in lines] +
                          [np.asarray([h[0] for h in hlines], dtype=float)]) if hlines else \
        np.concatenate([np.asarray(v, dtype=float) for _, _, v, _ in lines])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.10 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    tmax = max(float(t[-1]), 1e-3)
    for i in range(5):
        yy = py0 + (py1 - py0) * i / 4
        val = hi - (hi - lo) * i / 4
        draw.line((px0, yy, px1, yy), fill=GRID)
        if i < 4:
            text_r(draw, (px0 - 8, yy), yfmt.format(val), font(12), MUTED)
    for sec in range(int(math.floor(tmax)) + 1):
        xx = px0 + (px1 - px0) * (sec / tmax)
        draw.line((xx, py0, xx, py1), fill=GRID)
        draw.text((xx, py1 + 18), f"{sec}s", font=font(11), fill=MUTED, anchor="mt")
    if ylabel:
        draw.text((x0 + 8, y0 + 26), ylabel, font=font(11), fill=MUTED)
    for vline in vlines:
        xx = px0 + (px1 - px0) * (vline / tmax)
        dashed(draw, (xx, py0), (xx, py1), (58, 64, 76), dash=4, gap=5)
    for vline in red_vlines:
        xx = px0 + (px1 - px0) * (vline / tmax)
        dashed(draw, (xx, py0), (xx, py1), RED, dash=5, gap=4, width=2)
    for val, lab, col in hlines:
        yy = py1 - (py1 - py0) * (val - lo) / (hi - lo)
        dashed(draw, (px0, yy), (px1, yy), col, dash=7, gap=5)
        draw.text((px1 - 4, yy - 4), lab, font=font(11), fill=col, anchor="rb")
    for label, col, values, width in lines:
        pts = [(px0 + (px1 - px0) * (t[i] / tmax),
                py1 - (py1 - py0) * (float(values[i]) - lo) / (hi - lo))
               for i in range(len(t))]
        draw.line(pts, fill=col, width=width)
    lx, ly = px0 + 8, py0 + 6
    for label, col, _, _ in lines:
        draw.rectangle((lx, ly + 3, lx + 9, ly + 11), fill=col)
        draw.text((lx + 14, ly), label, font=font(12), fill=MUTED)
        ly += 16
    for note in notes:
        draw.text((x0 + 8, y1 - 34 + note[2]), note[0], font=font(12), fill=note[1])
    return lambda tt, vv: (px0 + (px1 - px0) * (tt / tmax),
                           py1 - (py1 - py0) * (vv - lo) / (hi - lo))


def marker(draw, xy, label, col, shape="o"):
    x, y = xy
    if shape == "o":
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=col, outline=(0, 0, 0))
    elif shape == "up":
        draw.polygon([(x, y - 6), (x - 5, y + 3), (x + 5, y + 3)], fill=col, outline=(0, 0, 0))
    elif shape == "down":
        draw.polygon([(x, y + 6), (x - 5, y - 3), (x + 5, y - 3)], fill=col, outline=(0, 0, 0))
    else:
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=col, width=2)
        draw.line((x - 4, y + 4, x + 4, y - 4), fill=col, width=2)
    if label:
        draw.text((x + 7, y - 6), label, font=font(11), fill=col)


def event_text(e):
    name = e["event"]
    if name == "start":
        return f"开始 {e.get('goal', '')}"
    if name == "primitive_end":
        return f"{e['prim']} 结束 -> {e.get('next', '')}（{e.get('status', '')}）"
    if name == "crossed_above_hold":
        return f"方块越过 hold_z：z={e['block_z']:.4f} m, grip={e['grip']:.3f}"
    if name == "crossed_below_hold":
        return f"方块跌破 hold_z：z={e['block_z']:.4f} m, grip={e['grip']:.3f}  ← 打滑"
    if name == "released":
        return f"释放：z={e.get('block_z', 0):.4f} m"
    if name == "dropped":
        return f"判定 dropped：z={e.get('block_z', 0):.4f} m"
    if name == "guard":
        return "护栏触发：" + ",".join(e.get("trips", []))
    if name == "done":
        return f"结束：success={e['success']} reason={e.get('reason', '')}"
    if name == "wait_start":
        return "进入等待稳定"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CN))
    ap.add_argument("--trial", type=int, required=True, help="1-based trial number")
    ap.add_argument("--trace", default="data/demo/traces.jsonl")
    ap.add_argument("--meta", default="data/demo/trace_meta.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None, help="also write this episode's rows as a flat CSV")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1400)
    args = ap.parse_args()

    rows, ep, meta = load_episode(args)
    st = stats(rows, ep, meta)
    t = st["t"]
    tmax = max(float(t[-1]), 1e-3)
    segments = phase_segments(ep, tmax)
    slip_lines = [e["t"] for e in ep["events"] if e["event"] == "crossed_below_hold"][:1]
    key = []
    below_count, released_seen = 0, False
    for e in ep["events"]:
        if e["event"] == "primitive_end" and e["prim"] == "闭合":
            key.append(f"{e['t']:.2f}s 闭合 {e['status']} (grip {st['grip_close']:.3f})")
        elif e["event"] == "crossed_below_hold":
            below_count += 1
            tag = "← 打滑" if below_count == 1 and not released_seen else (
                "（释放）" if released_seen else "（再次跌破）")
            key.append(f"{e['t']:.2f}s 跌破 hold (grip {e['grip']:.3f}) {tag}")
        elif e["event"] == "released":
            released_seen = True
            key.append(f"{e['t']:.2f}s 释放")
        elif e["event"] == "dropped":
            key.append(f"{e['t']:.2f}s dropped")
        elif e["event"] == "guard":
            key.append(f"{e['t']:.2f}s 护栏 {','.join(e.get('trips', []))}")
    key_line = "  |  ".join(key)
    grip_close_s = "—" if st["grip_close"] is None else f"{st['grip_close']:.3f}"
    if st["grip_close"] is None:
        slip_s = ("未提起" if st["max_z"] <= st["rest_z"] + 0.002
                  else f"z 回落 {st['slip_mm']:.1f} mm")
    elif st["max_z"] <= st["hold_z"]:
        slip_s = f"夹持下沉 {st['slip_mm']:.1f} mm（未过 hold_z）"
    else:
        slip_s = f"打滑 {st['slip_mm']:.1f} mm"
    out = args.out or f"data/demo/diagnostics/{args.source}_t{args.trial:02d}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    w, h = args.width, args.height
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    outcome = ep["events"][-1]
    ok = bool(outcome.get("success", False))
    col = GREEN if ok else RED
    reason = outcome.get("reason", "")
    draw.text((40, 18), f"{SOURCE_CN[args.source]} · 试次 {args.trial:02d}/20 · "
                        f"r={ep['radius']:.3f} m · θ={ep['theta_deg']:+.1f}° · "
                        f"定位误差 {ep['pose_error_mm']:.1f} mm",
              font=font(26), fill=TEXT)
    draw.text((40, 56), f"{'✓ 成功放置' if ok else '✗ ' + reason} · "
                        f"闭合结束 grip={grip_close_s} rad · 方块最高 {st['max_z']:.4f} m · "
                        f"{slip_s} · held {st['max_held']} ticks",
              font=font(17), fill=col)
    draw_phase_bar(draw, 60, w - 40, 96, 22, tmax, segments)
    vlines = [e["t"] for e in ep["events"] if e["event"] == "primitive_end"]

    gap, x0, x1 = 24, 60, w - 40
    pw = (x1 - x0 - gap) // 2
    top, ph = 132, 360
    boxes = [(x0 + (i % 2) * (pw + gap), top + (i // 2) * (ph + gap),
              x0 + (i % 2) * (pw + gap) + pw, top + (i // 2) * (ph + gap) + ph)
             for i in range(6)]

    mz = panel(draw, boxes[0], "方块高度 vs 末端高度（m）", t,
               [("末端 z", BLUE, st["eez"], 2), ("方块 z", ORANGE, st["z"], 2)],
               hlines=((st["hold_z"], "hold_z", MUTED), (st["rest_z"], "桌面", (150, 110, 80))),
               yfmt="{:.3f}", ylabel="m", vlines=vlines, red_vlines=slip_lines)
    off = 0
    for e in ep["events"]:
        if e["event"] == "crossed_above_hold":
            marker(draw, mz(e["t"], e["block_z"]), "", GREEN, "up")
        elif e["event"] == "crossed_below_hold":
            marker(draw, mz(e["t"], e["block_z"]), "", RED, "down")
        elif e["event"] == "dropped":
            marker(draw, mz(e["t"], e["block_z"]), "", RED, "o")
        elif e["event"] == "released":
            marker(draw, mz(e["t"], e["block_z"]), "", BLUE, "o")
        elif e["event"] == "guard":
            marker(draw, mz(e["t"], st["hold_z"]), "", RED, "x")

    panel(draw, boxes[1], "夹爪开度 q（rad）", t,
          [("实际 q", GREEN, st["grip"], 2), ("设定 q_target", MUTED, st["gtgt"], 1)],
          hlines=((CLOSE_Q, "CLOSE_Q", RED), (OPEN_Q, "OPEN_Q", BLUE)),
          yfmt="{:.2f}", ylabel="rad", vlines=vlines, red_vlines=slip_lines)

    panel(draw, boxes[2], "桌面接触力（N）", t,
          [("接触力", RED, st["force"], 2)],
          hlines=((TABLE_FORCE_LIMIT, f"护栏 {TABLE_FORCE_LIMIT:.0f} N", ORANGE),),
          yfmt="{:.2f}", ylabel="N", vlines=vlines, red_vlines=slip_lines)

    panel(draw, boxes[3], "臂关节最大力矩（N·m）", t,
          [("max |tau| (5 臂关节)", PURPLE, st["tau"], 2)],
          hlines=((TAU_LIMIT, f"堵转阈值 {TAU_LIMIT:.2f} N·m", ORANGE),),
          yfmt="{:.2f}", ylabel="N·m", vlines=vlines, red_vlines=slip_lines)

    panel(draw, boxes[4], "方块水平漂移（mm，相对初始位置）", t,
          [("|Δxy|", YELLOW, st["drift"], 2)],
          yfmt="{:.0f}", ylabel="mm", vlines=vlines, red_vlines=slip_lines)

    panel(draw, boxes[5], "方块竖直速度（m/s）", t,
          [("vz", CYAN, st["vz"], 2)],
          hlines=((0.0, "0", MUTED),),
          yfmt="{:+.2f}", ylabel="m/s", vlines=vlines, red_vlines=slip_lines)

    y = h - 104
    draw.text((40, y), f"闭合结束 grip={grip_close_s} rad · 方块最高 {st['max_z']:.4f} m · "
                       f"{slip_s} · 桌面力 max {st['max_force']:.2f} N · "
                       f"臂力矩 max {st['max_tau']:.2f} N·m · 最终漂移 {st['final_drift']:.1f} mm · "
                       f"held {st['max_held']} ticks",
              font=font(15), fill=TEXT)
    draw.text((40, y + 28), "关键事件：" + (key_line or "—"), font=font(13), fill=col)
    draw.text((40, y + 52), "每格 = 一个 50 Hz 决策 tick（20 ms）；灰虚线 = 原语切换；红虚线 = 第一次跌破 hold_z。"
                            "数据：data/demo/traces.jsonl",
              font=font(13), fill=MUTED)
    draw.text((40, y + 74), "方块 z 在末端 z 不变时下降 = 在夹爪里下滑；夹爪 q 停在 −0.22 附近 = 两颚合拢、方块已脱出。",
              font=font(13), fill=MUTED)
    img.save(out)

    print(f"wrote {out}")
    print(f"\n{'time':>7} {'tick':>5}  event")
    for e in ep["events"]:
        print(f"{e['t']:7.2f} {e['tick']:5d}  {event_text(e)}")
    print(f"\n闭合结束 grip={grip_close_s} rad, 方块最高 {st['max_z']:.4f} m, "
          f"打滑 {st['slip_mm']:.1f} mm, 桌面力 max {st['max_force']:.2f} N, "
          f"臂力矩 max {st['max_tau']:.2f} N·m, 最终漂移 {st['final_drift']:.1f} mm")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        vecs = {"q": 6, "q_target": 6, "qvel": 6, "tau": 6, "ee": 3, "ee_axis": 3,
                "block": 3, "block_quat": 4, "block_vel": 3}
        cols = (["t", "tick", "phase", "goal", "status", "grip", "table_force",
                 "lifted", "held_ticks", "done", "success", "reason"] +
                [f"{k}{i}" for k, n in vecs.items() for i in range(n)])
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(cols)
            for r in rows:
                flat = [r[c] for c in ("t", "tick", "phase", "goal", "status", "grip",
                                       "table_force", "lifted", "held_ticks", "done",
                                       "success", "reason")]
                for k, n in vecs.items():
                    flat.extend(r[k][:n])
                wr.writerow(flat)
        print(f"wrote {args.csv} ({len(rows)} rows x {len(cols)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
