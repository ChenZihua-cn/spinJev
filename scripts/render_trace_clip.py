#!/usr/bin/env python3
"""Side-by-side close-up of the same trial: single detection (drops) vs 20-frame average (places).

The wide 4-panel video is time-compressed; this clip starts at the close primitive, plays every
50 Hz decision tick as one video frame (0.5x), follows the block with the camera, and plots block z
against hold_z underneath. It is the "how did it slip / how did it hold" view.

Run:
    .venv/bin/python scripts/render_trace_clip.py
    .venv/bin/python scripts/render_trace_clip.py --trial 2 --left single --right avg20
"""

import argparse
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import render_pose_sources as rps  # noqa: E402

FONT_PATH = rps.FONT_PATH
FONT_INDEX = rps.FONT_INDEX
BG = (14, 16, 20)
TEXT = (228, 231, 237)
MUTED = (145, 151, 161)
GREEN = (92, 204, 124)
RED = (235, 92, 82)
ORANGE = (240, 150, 80)
BLUE = (84, 172, 232)


def font(size):
    return ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX)


def dashed(draw, p0, p1, fill, dash=5, gap=4, width=1):
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


def load_placement(trial0):
    meta = json.load(open("data/demo/trace_meta.json"))
    ep = next(e for e in meta["episodes"] if e["source"] == "single" and e["trial"] == trial0)
    other = next(e for e in meta["episodes"] if e["source"] == "avg20" and e["trial"] == trial0)
    if abs(ep["xy"][0] - other["xy"][0]) > 1e-9 or abs(ep["xy"][1] - other["xy"][1]) > 1e-9:
        raise SystemExit("the two sources were not run on the same placement")
    return ep, meta


class Panel:
    def __init__(self, source, trial0, xy, seed):
        self.source = source
        self.sim = rps.Sim(xml=rps.TABLE_SCENE)
        self.renderer = mujoco.Renderer(self.sim.model, height=390, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 140
        self.cam.elevation = -35
        self.cam.distance = 0.36
        self.runner = rps.Runner(source, self.sim, self.renderer, self.cam)
        self.runner.start(xy, trial0, seed)
        self.frames = []
        self.rows = []
        self.done_flags = []
        self.outcomes = []
        self._advance_to_close_start()
        self._capture()

    def _advance_to_close_start(self):
        while not any(e["event"] == "primitive_end" and e["prim"] == "下探"
                      for e in self.runner.events):
            if self.runner.finished:
                raise SystemExit(f"{self.source.key} finished before the close primitive")
            self.runner.step()
        self.slip_t = None

    def _capture(self):
        block = self.runner.sim.truth().obj_pos
        self.cam.lookat[:] = (block[0], block[1], max(0.055, float(block[2]) + 0.040))
        self.renderer.update_scene(self.runner.sim.data, camera=self.cam)
        self.frames.append(self.renderer.render().copy())
        self.rows.append(self.runner.rows[-1])
        self.done_flags.append(self.runner.finished)
        self.outcomes.append(self.runner.outcome_key if self.runner.finished else "")

    def step(self):
        if self.runner.finished:
            return
        self.runner.step()
        for e in self.runner.events:
            if e["event"] == "crossed_below_hold" and self.slip_t is None:
                self.slip_t = e["t"]
        if not self.runner.finished:
            self._capture()
        else:
            self._capture()

    @property
    def done(self):
        return self.runner.finished


def overlay(img, panel, strip_img, source_colour, i):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 640, 38), fill=(10, 12, 16))
    draw.rectangle((0, 352, 640, 390), fill=(10, 12, 16))
    draw.text((8, 5), panel.source.name, font=font(20), fill=source_colour)
    draw.text((632, 7), f"定位误差 {panel.runner.pose_err * 1000:.1f} mm",
              font=font(14), fill=TEXT, anchor="rt")
    row = panel.rows[i]
    draw.text((8, 355), f"{row['phase']} · z={row['block'][2]:.4f} m · "
                        f"grip={row['grip']:.3f} rad · F={row['table_force']:.2f} N · "
                        f"held {row['held_ticks']}",
              font=font(13), fill=TEXT)
    done = panel.done_flags[i]
    if done:
        ok = panel.outcomes[i] == "placed"
        col = GREEN if ok else RED
        fill = (16, 66, 34) if ok else (82, 22, 22)
        overlay_rgba = Image.new("RGBA", (640, 390), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay_rgba)
        od.rounded_rectangle((150, 130, 490, 210), radius=10, fill=fill + (225,),
                             outline=col + (255,), width=2)
        im = Image.alpha_composite(im.convert("RGBA"), overlay_rgba).convert("RGB")
        draw = ImageDraw.Draw(im)
        outcome = panel.outcomes[i]
        label = "✓ 成功放置" if ok else "× " + rps.reason_text(outcome)
        draw.text((320, 158), label, font=font(28), fill=col, anchor="mm")
        draw.text((320, 190), outcome,
                  font=font(13), fill=(230, 180, 175) if not ok else (150, 230, 175),
                  anchor="mm")
    # paste the z strip below the scene
    out = Image.new("RGB", (640, 390 + strip_img.height), BG)
    out.paste(im, (0, 0))
    out.paste(strip_img, (0, 390))
    return out


def strip(panel, total_frames, i, width=640, height=76):
    img = Image.new("RGB", (width, height), (17, 20, 24))
    draw = ImageDraw.Draw(img)
    x0, x1 = 46, width - 10
    y0, y1 = 8, height - 18
    zlo, zhi = 0.020, 0.080
    def px(i):
        return x0 + (x1 - x0) * (i / max(total_frames - 1, 1))
    def py(z):
        return y1 - (y1 - y0) * (z - zlo) / (zhi - zlo)
    draw.text((4, 4), "z", font=font(12), fill=MUTED)
    draw.text((x1, 4), "方块 z（m）", font=font(11), fill=MUTED, anchor="rt")
    for z, lab, col in ((rps.SPEC.min_wrist_z, None, None), (0.045, "hold_z", MUTED),
                        (0.030, "桌面", (150, 110, 80))):
        if not (zlo <= z <= zhi):
            continue
        y = py(z)
        dashed(draw, (x0, y), (x1, y), col, dash=5, gap=4)
        if lab:
            draw.text((x1 - 2, y - 3), lab, font=font(10), fill=col, anchor="rb")
    rows = panel.rows[: i + 1]
    t = [p["t"] for p in rows]
    z = [p["block"][2] for p in rows]
    if len(t) > 1:
        slip_idx = None
        if panel.slip_t is not None:
            slip_idx = next((k for k, tt in enumerate(t) if tt >= panel.slip_t), None)
        for k in range(1, len(t)):
            col = RED if (slip_idx is not None and k >= slip_idx) else ORANGE
            draw.line((px(k - 1), py(z[k - 1]), px(k), py(z[k])), fill=col, width=2)
        if slip_idx is not None:
            draw.ellipse((px(slip_idx) - 3, py(z[slip_idx]) - 3,
                          px(slip_idx) + 3, py(z[slip_idx]) + 3), fill=RED)
            draw.text((px(slip_idx) + 5, py(z[slip_idx]) - 12), "打滑起点",
                      font=font(10), fill=RED)
    draw.text((x0, height - 15), f"t={t[-1]:.2f}s / {t[-1] - t[0]:.2f}s after close start",
              font=font(10), fill=MUTED)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=2, help="1-based trial (default: 2, the A/B drop)")
    ap.add_argument("--left", default="single")
    ap.add_argument("--right", default="avg20")
    ap.add_argument("--out", default="data/demo/diagnostics/drop_vs_success.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--hold", type=float, default=1.6)
    args = ap.parse_args()

    trial0 = args.trial - 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ep, meta = load_placement(trial0)
    xy = np.array(ep["xy"])
    sources = {s.key: s for s in rps.SOURCES}
    panels = [
        Panel(sources[args.left], trial0, xy, 1000 + trial0),
        Panel(sources[args.right], trial0, xy, 1000 + trial0),
    ]
    while not all(p.done for p in panels):
        for p in panels:
            p.step()
    n = max(len(p.frames) for p in panels)
    for p in panels:
        while len(p.frames) < n:
            p.frames.append(p.frames[-1])
            p.rows.append(p.rows[-1])
            p.done_flags.append(p.done_flags[-1])
            p.outcomes.append(p.outcomes[-1])
    print(f"aligned {n} frames from close start "
          f"({panels[0].source.key} {len(panels[0].frames)}, {panels[1].source.key} {len(panels[1].frames)})")
    print(f"slip times: {panels[0].source.key}={panels[0].slip_t}, "
          f"{panels[1].source.key}={panels[1].slip_t}")

    scene_w, scene_h, strip_h = 640, 390, 76
    gap, margin, header, footer = 8, 10, 70, 46
    W = 2 * margin + 2 * scene_w + gap
    H = header + margin + scene_h + strip_h + margin + footer
    W += W % 2
    H += H % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def compose(i):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw.text((margin + 4, 8), "同一次抓取 · 单次检测 vs 悬停平均 20 帧",
                  font=font(24), fill=TEXT)
        draw.text((margin + 4, 38),
                  f"试次 {args.trial:02d}/20 · r={ep['radius']:.3f} m · θ={ep['theta_deg']:+.1f}° · "
                  f"左 {panels[0].runner.pose_err * 1000:.1f} mm / 右 {panels[1].runner.pose_err * 1000:.1f} mm · "
                  f"慢放 0.5×（每帧 = 一个 50 Hz tick）",
                  font=font(13), fill=MUTED)
        draw.line((0, header - 1, W, header - 1), fill=(40, 44, 52))
        for k, p in enumerate(panels):
            x = margin + k * (scene_w + gap)
            y = header + margin
            panel_img = overlay(p.frames[i], p, strip(p, n, i), p.source.colour, i)
            img.paste(panel_img, (x, y))
        y = H - footer
        draw.text((margin + 4, y + 5),
                  "方块 z 在末端不动时下降 = 在夹爪里下滑；夹爪 q 停在 ~0.28 rad = 方块还在两颚之间，"
                  "降到 −0.22 = 两颚合拢、方块已脱出。",
                  font=font(12), fill=MUTED)
        draw.text((margin + 4, y + 25),
                  "逐 tick 数据：data/demo/traces.jsonl；同一试次，只有目标位姿来源不同。",
                  font=font(12), fill=MUTED)
        return img

    for i in range(n):
        proc.stdin.write(compose(i).tobytes())
    last = compose(n - 1)
    for _ in range(int(args.hold * args.fps)):
        proc.stdin.write(last.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")
    for p in panels:
        p.renderer.close()
    print(f"wrote {args.out} ({W}x{H}, {n} frames + hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
