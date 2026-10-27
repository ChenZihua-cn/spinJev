#!/usr/bin/env python3
"""Render the four pose-source sweep as one MP4: same script, same 20 placements, four detectors.

The table this visualises lives in README "真正的杠杆是感知通道，不是机械". Four panels run in
lockstep: the oracle (diagnostic only), a single noisy detection, and 20- / 60-frame hover averages.
Each trial uses the same placement for all four panels, and each panel draws its own detection
noise per placement -- reusing one noise realisation across all 20 trials is what made the old
table's single-detection row look like a cliff.

Run:
    .venv/bin/python scripts/render_pose_sources.py                 # full 20-trial video
    .venv/bin/python scripts/render_pose_sources.py --no-video      # numbers only, no EGL
    .venv/bin/python scripts/render_pose_sources.py --trials 3      # short preview

Outputs (data/ is gitignored):
    data/demo/pose_sources.mp4
    data/demo/pose_sources.json     # placements + every episode outcome, for recomputing rates
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

# Must be set before importing mujoco. EGL works on this machine; on the cluster use osmesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # --no-video does not need the overlay library
    Image = ImageDraw = ImageFont = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from spinjev.backends.sim.mujoco_sim import TABLE_SCENE, Sim  # noqa: E402
from spinjev.perception.simulator import Perceptor  # noqa: E402
from spinjev.spine.primitives import Primitive  # noqa: E402
from spinjev.tasks.pick_place import (  # noqa: E402
    DROP_ZONE,
    R_NEAR,
    R_REACH,
    THETA_MAX,
    PickPlace,
    TaskSpec,
)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_INDEX = 2  # Simplified Chinese face inside the .ttc

SPEC = TaskSpec(grasp_depth=0.090, grasp_lateral=-0.030, min_wrist_z=0.115, lift_height=0.15)
AIM_PHASES = ("回位", "悬停", "下探", "闭合", "提升", "移送", "释放")


class Source:
    def __init__(self, key, name, detail, sigma, aim_ticks, colour):
        self.key = key
        self.name = name
        self.detail = detail
        self.sigma = float(sigma)
        self.aim_ticks = int(aim_ticks)
        self.colour = colour


SOURCES = (
    Source("truth", "真值位姿", "σ=0 · 仅诊断，不进入观测", 0.0, 1, (205, 205, 210)),
    Source("single", "单次检测", "σ=10 mm/轴 · 单帧", 0.010, 1, (235, 92, 82)),
    Source("avg20", "悬停平均 20 帧", "= 对准 · 部署配置", 0.010, 20, (92, 204, 124)),
    Source("avg60", "悬停平均 60 帧", "σ 压到 ~1.3 mm/轴", 0.010, 60, (84, 172, 232)),
)

REASON_CN = {
    "table_hit": "撞桌",
    "arm_straining": "关节堵转",
    "dropped": "滑落",
    "timeout": "超时",
    "never gripped": "未夹住",
    "held": "未提升",
}


def reason_text(reason):
    if not reason:
        return "未完成"
    raw = reason.replace("guard:", "")
    return " ".join(REASON_CN.get(p, p) for p in raw.split(",") if p)


def rlist(v, n=5):
    return [round(float(x), n) for x in np.asarray(v).ravel()]


def placements(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r, th = rng.uniform(R_NEAR, R_REACH), rng.uniform(-THETA_MAX, THETA_MAX)
        out.append((r, th, np.array([r * math.sin(th), -r * math.cos(th)])))
    return out


class Runner:
    """One panel: one sim, one detector, and the scripted sequence as a resumable state machine."""

    def __init__(self, source, sim, renderer, camera):
        self.source = source
        self.sim = sim
        self.renderer = renderer
        self.camera = camera
        self.perc = Perceptor(
            sigma_clear=source.sigma, sigma_occluded=source.sigma, miss_rate=0.0, seed=3
        )
        self.env = PickPlace(sim=sim, perceptor=self.perc, spec=SPEC, seed=3)
        self.last_img = None
        self.ticks = 0
        self.pose_err = 0.0
        self.goals = []
        self.idx = 0
        self.phase = "drive"
        self.wait_left = 0
        self.finished = False
        self.trial = 0
        self.n = 1
        self.radius = 0.0
        self.theta = 0.0
        self.successes_before = 0
        self.completed_before = 0
        self.rows = []
        self.events = []
        self.percept = None
        self.lifted_before = False
        self.released_before = False
        self.dropped_before = False
        self.tripped_before = ()
        self.done_before = False

    def start(self, xy, trial, percept_seed):
        self.env.reset(xy=xy, unreachable=False, occluded=False)
        self.perc.reset(seed=percept_seed)  # fresh noise for every placement, not one draw repeated
        p = self.env.aim(self.source.aim_ticks)
        self.pose_err = float(np.linalg.norm(p.pos - self.env.truth().obj_pos))
        th = self.env.heading_of(p.pos[:2])
        self.goals = [
            (Primitive.APPROACH, SPEC.home),
            (Primitive.APPROACH, self.env.wrist_goal(p.pos[:2], p.pos[2], th, lift=SPEC.pregrasp_clear)),
            (Primitive.APPROACH, self.env.wrist_goal(p.pos[:2], p.pos[2], th)),
            (Primitive.CLOSE, None),
            (Primitive.LIFT, [0.0, 0.0, SPEC.lift_height]),
            (Primitive.APPROACH, [DROP_ZONE[0], DROP_ZONE[1], SPEC.lift_height]),
            (Primitive.RELEASE, None),
        ]
        self.idx = 0
        self.phase = "drive"
        self.wait_left = 0
        self.finished = False
        self.ticks = 0
        self.rows = []
        self.events = []
        self.percept = p.pos.copy()
        self.lifted_before = False
        self.released_before = False
        self.dropped_before = False
        self.tripped_before = ()
        self.done_before = False
        self.env.spine.command(*self.goals[0])
        self.event("start", goal=AIM_PHASES[0], pose_error_mm=round(self.pose_err * 1000, 2))
        self.record()
        self.render()

    def render(self):
        if self.renderer is None:
            return
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        self.last_img = self.renderer.render().copy()

    def event(self, name, **fields):
        self.events.append({"t": round(float(self.sim.data.time), 4), "tick": int(self.ticks),
                            "event": name, **fields})

    def record(self):
        d = self.sim.data
        sense = self.sim.sense()
        truth = self.sim.truth()
        out = self.env.out
        q = d.qpos[self.sim.arm_qadr]
        self.rows.append({
            "t": round(float(d.time), 4),
            "tick": int(self.ticks),
            "phase": self.phase_label,
            "goal": (AIM_PHASES[self.idx] if self.phase == "drive" else "wait"),
            "status": self.env.spine.status,
            "q": rlist(q),
            "q_target": rlist(self.env.spine.q_target),
            "qvel": rlist(d.qvel[self.sim.arm_dofadr], 4),
            "tau": rlist(d.actuator_force[self.sim.ujnt], 4),
            "ee": rlist(sense.ee_pos),
            "ee_axis": rlist(sense.ee_axis, 4),
            "block": rlist(truth.obj_pos),
            "block_quat": rlist(truth.obj_quat, 4),
            "block_vel": rlist(truth.obj_vel, 4),
            "grip": round(float(q[5]), 5),
            "table_force": round(float(self.sim.contact_force_with_table()), 4),
            "trips": list(out.tripped),
            "lifted": bool(float(truth.obj_pos[2]) > self.env.hold_z),
            "held_ticks": int(out.held_ticks),
            "done": bool(out.done),
            "success": bool(out.success),
            "reason": out.reason or "",
        })

    def step(self):
        if self.finished:
            return
        idx0, phase0 = self.idx, self.phase
        done0 = self.env.out.done
        trips0 = tuple(self.env.out.tripped)
        block_z0 = float(self.sim.truth().obj_pos[2])
        released0 = self.env.out.released
        dropped0 = self.env.out.dropped
        end_status = ""
        if self.phase == "wait":
            self.env.spine.tick()
            self.env.sim.step(self.env.spine.decim)
            self.env.update()
            self.wait_left -= 1
            if self.env.out.done or self.wait_left <= 0:
                self.finished = True
        else:
            if self.env.spine.status == "running" and not self.env.out.done:
                self.env.spine.advance(1)
                self.env.update()
            if self.env.out.done:
                self.finished = True
            elif self.env.spine.status != "running":
                end_status = self.env.spine.status
                self.idx += 1
                if self.idx < len(self.goals):
                    self.env.spine.command(*self.goals[self.idx])
                else:
                    self.phase = "wait"
                    self.wait_left = 120
        self.ticks += 1
        if self.finished:
            self.render()
        if idx0 != self.idx:
            self.event("primitive_end", prim=AIM_PHASES[min(idx0, len(AIM_PHASES) - 1)],
                       status=end_status,
                       next=(AIM_PHASES[min(self.idx, len(AIM_PHASES) - 1)]
                             if self.phase == "drive" else "wait"))
        if phase0 == "drive" and self.phase == "wait":
            self.event("wait_start")
        block_z1 = float(self.sim.truth().obj_pos[2])
        if block_z1 > self.env.hold_z and block_z0 <= self.env.hold_z:
            self.event("crossed_above_hold", block_z=round(block_z1, 5),
                       grip=round(float(self.sim.data.qpos[self.sim.gadr]), 4))
        if block_z1 <= self.env.hold_z and block_z0 > self.env.hold_z and not self.env.out.released:
            self.event("crossed_below_hold", block_z=round(block_z1, 5),
                       grip=round(float(self.sim.data.qpos[self.sim.gadr]), 4))
        if self.env.out.released and not released0:
            self.event("released", block_z=round(block_z1, 5))
        if self.env.out.dropped and not dropped0:
            self.event("dropped", block_z=round(block_z1, 5))
        if tuple(self.env.out.tripped) != trips0:
            self.event("guard", trips=list(self.env.out.tripped))
        if self.env.out.done and not done0:
            self.event("done", success=bool(self.env.out.success),
                       reason=self.env.out.reason or "")
        self.record()

    @property
    def phase_label(self):
        if self.finished:
            return "结束"
        if self.phase == "wait":
            return "等待稳定"
        return AIM_PHASES[min(self.idx, len(AIM_PHASES) - 1)]

    @property
    def outcome_key(self):
        if not self.finished:
            return ""
        out = self.env.out
        if out.success:
            return "placed"
        return out.reason or ("held" if out.held_ticks else "never gripped")

    @property
    def outcome_text(self):
        key = self.outcome_key
        if key == "placed":
            return "✓ 成功放置"
        return f"✗ {reason_text(key)}"


class Canvas:
    def __init__(self, panel_w, panel_h):
        if Image is None:
            raise SystemExit("Pillow is required for video output (or pass --no-video)")
        self.pw, self.ph = panel_w, panel_h
        self.s = panel_w / 480.0
        self.margin = max(6, int(round(10 * self.s)))
        self.gap = max(4, int(round(8 * self.s)))
        self.header = max(44, int(round(70 * self.s)))
        self.footer = max(28, int(round(44 * self.s)))
        self.w = 2 * self.margin + 2 * panel_w + self.gap
        self.h = self.header + self.margin + 2 * panel_h + self.gap + self.margin + self.footer
        self.w += self.w % 2
        self.h += self.h % 2
        self.fonts = {}

    def font(self, size):
        size = max(8, int(round(size * self.s)))
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX)
        return self.fonts[size]

    def base(self):
        return Image.new("RGB", (self.w, self.h), (14, 16, 20))

    def header_bar(self, draw, trial, n):
        s = self.s
        draw.text((self.margin + int(4 * s), int(9 * s)),
                  "四个位姿来源 · 同一套脚本 · 同 20 个可达位点",
                  font=self.font(24), fill=(236, 239, 243))
        draw.text((self.margin + int(4 * s), int(40 * s)),
                  "SO-ARM100 / MuJoCo " + mujoco.__version__ +
                  " · 500 Hz 伺服 / 50 Hz 决策 · 手调脚本，零学习成分",
                  font=self.font(13), fill=(150, 156, 166))
        draw.text((self.w - self.margin - int(4 * s), int(10 * s)), f"试次 {trial + 1:02d}/{n:02d}",
                  font=self.font(18), fill=(236, 239, 243), anchor="rt")
        draw.line((0, self.header - 1, self.w, self.header - 1), fill=(40, 44, 52), width=1)

    def footer_bar(self, draw, trial, n):
        s = self.s
        y = self.h - self.footer
        draw.text((self.margin + int(4 * s), y + int(6 * s)),
                  "每点位独立采样检测噪声 · 真值仅诊断 · n=20 只用于找杠杆，不当成功率发表",
                  font=self.font(12), fill=(140, 146, 156))
        bar_x = self.margin + int(4 * s)
        bar_y = self.h - int(12 * s)
        bar_w = self.w - 2 * bar_x
        bar_h = max(3, int(round(5 * s)))
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=(40, 44, 52))
        frac = (trial + 1) / max(n, 1)
        draw.rectangle((bar_x, bar_y, bar_x + int(bar_w * frac), bar_y + bar_h), fill=(92, 204, 124))

    def panel(self, base, col, row, runner):
        s = self.s
        x = self.margin + col * (self.pw + self.gap)
        y = self.header + self.margin + row * (self.ph + self.gap)
        if runner.last_img is not None:
            base.paste(Image.fromarray(runner.last_img), (x, y))
        else:
            base.paste(Image.new("RGB", (self.pw, self.ph), (24, 27, 33)), (x, y))

        draw = ImageDraw.Draw(base)
        top = int(round(46 * s))
        bottom = int(round(28 * s))
        draw.rectangle((x, y, x + self.pw, y + top), fill=(10, 12, 16))
        draw.rectangle((x, y + self.ph - bottom, x + self.pw, y + self.ph), fill=(10, 12, 16))
        draw.text((x + int(8 * s), y + int(5 * s)), runner.source.name,
                  font=self.font(19), fill=runner.source.colour)
        draw.text((x + int(8 * s), y + int(27 * s)), runner.source.detail,
                  font=self.font(12), fill=(150, 156, 166))
        done = runner.finished
        draw.text((x + self.pw - int(8 * s), y + int(7 * s)),
                  "✓ " + str(runner.successes_before + (1 if done and runner.env.out.success else 0)),
                  font=self.font(17), fill=(92, 204, 124), anchor="rt")
        draw.text((x + self.pw - int(8 * s), y + int(28 * s)),
                  f"完成 {runner.completed_before + (1 if done else 0)}",
                  font=self.font(11), fill=(150, 156, 166), anchor="rt")
        draw.text((x + int(8 * s), y + self.ph - int(6 * s)),
                  f"试次 {runner.trial + 1:02d}/{runner.n:02d} · r={runner.radius:.2f} m · "
                  f"θ={math.degrees(runner.theta):+.0f}°",
                  font=self.font(12), fill=(210, 214, 222), anchor="ld")
        draw.text((x + self.pw // 2, y + self.ph - int(6 * s)), f"当前：{runner.phase_label}",
                  font=self.font(12), fill=runner.source.colour, anchor="md")
        draw.text((x + self.pw - int(8 * s), y + self.ph - int(6 * s)),
                  f"误差 {runner.pose_err * 1000:.1f} mm",
                  font=self.font(12), fill=(230, 232, 238), anchor="rd")

        if done:
            ok = runner.env.out.success
            fill = (16, 66, 34) if ok else (82, 22, 22)
            edge = (82, 200, 116) if ok else (235, 92, 82)
            box_w, box_h = int(round(330 * s)), int(round(78 * s))
            bx, by = x + (self.pw - box_w) // 2, y + int(round(136 * s))
            overlay = Image.new("RGBA", (self.pw, self.ph), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rounded_rectangle((bx - x, by - y, bx - x + box_w, by - y + box_h),
                                 radius=max(4, int(round(10 * s))), fill=fill + (225,),
                                 outline=edge + (255,), width=max(1, int(round(2 * s))))
            base.paste(Image.alpha_composite(
                base.crop((x, y, x + self.pw, y + self.ph)).convert("RGBA"), overlay).convert("RGB"),
                (x, y))
            draw = ImageDraw.Draw(base)
            if ok:
                draw.text((x + self.pw // 2, y + int(round(158 * s))), "✓ 成功放置",
                          font=self.font(28), fill=(120, 230, 150), anchor="mm")
            else:
                key = runner.outcome_key
                draw.text((x + self.pw // 2, y + int(round(157 * s))), f"× {reason_text(key)}",
                          font=self.font(25), fill=(245, 120, 110), anchor="mm")
                draw.text((x + self.pw // 2, y + int(round(188 * s))), key,
                          font=self.font(13), fill=(230, 180, 175), anchor="mm")


def compose(canvas, trial, n, runners):
    img = canvas.base()
    draw = ImageDraw.Draw(img)
    canvas.header_bar(draw, trial, n)
    canvas.footer_bar(draw, trial, n)
    for i, runner in enumerate(runners):
        canvas.panel(img, i % 2, i // 2, runner)
    return img


def card_intro(canvas):
    img = canvas.base()
    draw = ImageDraw.Draw(img)
    w, h = canvas.w, canvas.h
    draw.text((w // 2, int(h * 0.14)), "四个位姿来源 · 同一套脚本 · 同 20 个可达位点",
              font=canvas.font(30), fill=(236, 239, 243), anchor="mt")
    draw.text((w // 2, int(h * 0.14) + int(46 * canvas.s)), "只换目标位姿来源，其他全部不变",
              font=canvas.font(18), fill=(150, 156, 166), anchor="mt")
    y = int(h * 0.34)
    for i, source in enumerate(SOURCES):
        x = int(w * (0.055 + i * 0.235))
        draw.rectangle((x, y, x + int(10 * canvas.s), y + int(10 * canvas.s)),
                       fill=source.colour)
        draw.text((x + int(20 * canvas.s), y - int(4 * canvas.s)), source.name,
                  font=canvas.font(20), fill=source.colour)
        draw.text((x + int(20 * canvas.s), y + int(24 * canvas.s)), source.detail,
                  font=canvas.font(13), fill=(150, 156, 166))
    notes = (
        "检测噪声 σ=10 mm/轴 · 每个位点独立采样",
        "同一套脚本序列、同一标定；真值位姿仅用于诊断，不进入观测（PLAN 2.7）",
        "n=20 只够找杠杆，不当成功率发表 · 手调脚本，零学习成分",
    )
    for i, note in enumerate(notes):
        draw.text((w // 2, int(h * 0.62) + i * int(30 * canvas.s)), note,
                  font=canvas.font(15), fill=(190, 195, 205), anchor="mt")
    return img


def summary_rows(results):
    rows = []
    for source in SOURCES:
        r = results[source.key]
        counts = {}
        for ep in r["episodes"]:
            if not ep["success"]:
                label = reason_text(ep["reason"])
                counts[label] = counts.get(label, 0) + 1
        fail = " ".join(f"{label}{count}" for label, count in
                        sorted(counts.items(), key=lambda kv: -kv[1])) or "—"
        err = float(np.mean([ep["pose_error_mm"] for ep in r["episodes"]]))
        rows.append((source, r["successes"], len(r["episodes"]), fail, err))
    return rows


def card_summary(canvas, results):
    img = canvas.base()
    draw = ImageDraw.Draw(img)
    w, h = canvas.w, canvas.h
    rows = summary_rows(results)
    draw.text((w // 2, int(h * 0.07)), "结果（每个位点独立采样检测噪声）",
              font=canvas.font(30), fill=(236, 239, 243), anchor="mt")
    draw.text((w // 2, int(h * 0.07) + int(44 * canvas.s)),
              "20 个随机可达位点 · 同一套脚本 · 只换目标位姿来源",
              font=canvas.font(16), fill=(150, 156, 166), anchor="mt")
    x0, x1, x2, x3 = int(w * 0.12), int(w * 0.40), int(w * 0.54), int(w * 0.76)
    y = int(h * 0.20)
    draw.text((x0, y), "位姿来源", font=canvas.font(14), fill=(130, 136, 146))
    draw.text((x1, y), "成功", font=canvas.font(14), fill=(130, 136, 146))
    draw.text((x2, y), "失败构成", font=canvas.font(14), fill=(130, 136, 146))
    draw.text((x3, y), "平均定位误差", font=canvas.font(14), fill=(130, 136, 146))
    draw.line((x0, y + int(24 * canvas.s), w - x0, y + int(24 * canvas.s)),
              fill=(50, 54, 62), width=1)
    row_h = int(58 * canvas.s)
    for i, (source, ok, n, fail, err) in enumerate(rows):
        yy = y + int(44 * canvas.s) + i * row_h
        draw.text((x0, yy), source.name, font=canvas.font(21), fill=source.colour)
        draw.text((x1, yy), f"{ok}/{n}", font=canvas.font(23), fill=(236, 239, 243))
        draw.text((x2, yy + int(3 * canvas.s)), fail, font=canvas.font(15), fill=(200, 204, 212))
        draw.text((x3, yy + int(3 * canvas.s)), f"{err:.1f} mm",
                  font=canvas.font(16), fill=(200, 204, 212))
    notes = (
        "真值位姿仅用于诊断；观测通路不读真值（PLAN 2.7）。",
        "旧表的单次 2/20 是同一噪声实现重复 20 次；本片每个位点独立采样。",
        "n=20 只够找杠杆，不当成功率发表。手调脚本，零学习成分。",
    )
    ny = y + int(44 * canvas.s) + 4 * row_h + int(26 * canvas.s)
    for i, note in enumerate(notes):
        draw.text((x0, ny + i * int(28 * canvas.s)), note,
                  font=canvas.font(13), fill=(140, 146, 156))
    draw.text((w // 2, h - int(30 * canvas.s)), "每个数字都能从 pose_sources.json 复算",
              font=canvas.font(13), fill=(120, 126, 136), anchor="mt")
    return img


def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.01, -0.27, 0.10)
    cam.distance = 0.66
    cam.azimuth = 140
    cam.elevation = -30
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/pose_sources.mp4")
    ap.add_argument("--log", default="data/demo/pose_sources.json")
    ap.add_argument("--trials", type=int, default=20, help="number of placements (max 20)")
    ap.add_argument("--seed", type=int, default=5, help="placement sampler seed")
    ap.add_argument("--percept-seed", type=int, default=1000, help="first per-trial detector seed")
    ap.add_argument("--render-every", type=int, default=6, help="decision ticks per rendered frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640, help="panel width")
    ap.add_argument("--height", type=int, default=480, help="panel height")
    ap.add_argument("--hold", type=float, default=0.6, help="seconds frozen after each trial")
    ap.add_argument("--intro", type=float, default=2.5)
    ap.add_argument("--summary", type=float, default=7.0)
    ap.add_argument("--no-video", action="store_true", help="run the sweep without EGL/ffmpeg")
    ap.add_argument("--trace-out", default="data/demo/traces.jsonl")
    ap.add_argument("--trace-meta", default="data/demo/trace_meta.json")
    ap.add_argument("--no-trace", action="store_true", help="skip the per-tick trace")
    args = ap.parse_args()

    n = min(int(args.trials), 20)
    places = placements(args.seed, n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    trace_fh = None
    trace_meta = []
    if not args.no_trace:
        os.makedirs(os.path.dirname(args.trace_out) or ".", exist_ok=True)
        trace_fh = open(args.trace_out, "w")

    canvas = None
    renderers = []
    ffmpeg = None
    if not args.no_video:
        canvas = Canvas(args.width, args.height)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{canvas.w}x{canvas.h}", "-r", str(args.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", args.out,
        ]
        ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def emit(img):
        if ffmpeg is not None:
            ffmpeg.stdin.write(img.tobytes())

    camera = make_camera()
    runners = []
    for source in SOURCES:
        sim = Sim(xml=TABLE_SCENE)
        renderer = None
        if not args.no_video:
            renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
            renderers.append(renderer)
        runners.append(Runner(source, sim, renderer, camera))

    if canvas is not None:
        intro = card_intro(canvas)
        for _ in range(int(args.intro * args.fps)):
            emit(intro)

    results = {s.key: {"successes": 0, "episodes": []} for s in SOURCES}
    hold_frames = max(1, int(args.hold * args.fps))
    t_start = time.perf_counter()

    for trial, (r, theta, xy) in enumerate(places):
        for runner in runners:
            runner.trial, runner.n = trial, n
            runner.radius, runner.theta = r, theta
            runner.successes_before = results[runner.source.key]["successes"]
            runner.completed_before = trial
            runner.start(xy, trial, args.percept_seed + trial)

        tick = 0
        while not all(runner.finished for runner in runners):
            for runner in runners:
                runner.step()
            tick += 1
            if canvas is not None and (tick % args.render_every == 0
                                       or all(runner.finished for runner in runners)):
                emit(compose(canvas, trial, n, runners))

        for runner in runners:
            out = runner.env.out
            key = runner.outcome_key
            results[runner.source.key]["successes"] += int(out.success)
            results[runner.source.key]["episodes"].append({
                "trial": trial,
                "radius": round(float(r), 4),
                "theta_deg": round(float(math.degrees(theta)), 2),
                "xy": [round(float(v), 4) for v in xy],
                "pose_error_mm": round(float(runner.pose_err * 1000), 2),
                "success": bool(out.success),
                "reason": "" if out.success else key,
                "ticks": int(out.ticks),
                "held_ticks": int(out.held_ticks),
                "reward": round(float(out.reward), 3),
                "wall_ticks": int(runner.ticks),
            })
            print(f"  trial {trial + 1:02d}/{n} {runner.source.key:6s} "
                  f"r={r:.3f} th={math.degrees(theta):+5.1f} "
                  f"err={runner.pose_err * 1000:5.1f}mm "
                  f"{'OK' if out.success else key}")
        if trace_fh is not None:
            for runner in runners:
                for row in runner.rows:
                    trace_fh.write(json.dumps({"source": runner.source.key, "trial": trial, **row},
                                              ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_meta.append({
                    "source": runner.source.key,
                    "trial": trial,
                    "radius": round(float(r), 4),
                    "theta_deg": round(float(math.degrees(theta)), 2),
                    "xy": [round(float(v), 4) for v in xy],
                    "percept": rlist(runner.percept),
                    "pose_error_mm": round(float(runner.pose_err * 1000), 2),
                    "aim_ticks": runner.source.aim_ticks,
                    "events": runner.events,
                    "rows": len(runner.rows),
                })
        final = compose(canvas, trial, n, runners) if canvas is not None else None
        if final is not None:
            for _ in range(hold_frames):
                emit(final)

    if canvas is not None:
        summary = card_summary(canvas, results)
        for _ in range(int(args.summary * args.fps)):
            emit(summary)
        ffmpeg.stdin.close()
        rc = ffmpeg.wait()
        if rc != 0:
            raise SystemExit(f"ffmpeg exited {rc}")
        for renderer in renderers:
            renderer.close()

    if trace_fh is not None:
        trace_fh.close()
        with open(args.trace_meta, "w") as fh:
            json.dump({
                "mujoco": mujoco.__version__,
                "backend": os.environ.get("MUJOCO_GL"),
                "protocol": {
                    "decision_hz": 50,
                    "placement_seed": args.seed,
                    "detector_seed_base": args.percept_seed,
                    "detector_seed_rule": "base + trial (fresh noise every placement)",
                    "task_spec": {
                        "grasp_depth": SPEC.grasp_depth,
                        "grasp_lateral": SPEC.grasp_lateral,
                        "min_wrist_z": SPEC.min_wrist_z,
                        "lift_height": SPEC.lift_height,
                        "home": [float(v) for v in SPEC.home],
                        "max_ticks": SPEC.max_ticks,
                    },
                    "hold_z": round(float(runners[0].env.hold_z), 5),
                    "block_rest_z": round(float(runners[0].env.block_z), 5),
                },
                "episodes": trace_meta,
            }, fh, ensure_ascii=False, indent=2)

    log = {
        "mujoco": mujoco.__version__,
        "backend": os.environ.get("MUJOCO_GL"),
        "protocol": {
            "placements": "uniform r in [R_NEAR, R_REACH], theta in [-THETA_MAX, THETA_MAX]",
            "placement_seed": args.seed,
            "detector_seed_base": args.percept_seed,
            "detector_seed_rule": "base + trial (fresh noise every placement)",
            "detector_sigma_m_per_axis": 0.010,
            "aim_ticks": {s.key: s.aim_ticks for s in SOURCES},
            "task_spec": {
                "grasp_depth": SPEC.grasp_depth,
                "grasp_lateral": SPEC.grasp_lateral,
                "min_wrist_z": SPEC.min_wrist_z,
                "lift_height": SPEC.lift_height,
                "home": [float(v) for v in SPEC.home],
                "max_ticks": SPEC.max_ticks,
            },
            "hold_z": round(float(runners[0].env.hold_z), 5),
            "block_rest_z": round(float(runners[0].env.block_z), 5),
        },
        "placements": [{"radius": round(float(r), 4),
                        "theta_deg": round(float(math.degrees(t)), 2),
                        "xy": [round(float(v), 4) for v in xy]} for r, t, xy in places],
        "sources": {
            s.key: {
                "name": s.name,
                "successes": results[s.key]["successes"],
                "n": len(results[s.key]["episodes"]),
                "episodes": results[s.key]["episodes"],
            } for s in SOURCES
        },
    }
    with open(args.log, "w") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)

    print(f"\nresolution: {canvas.w}x{canvas.h} @ {args.fps} fps" if canvas else "\nheadless run")
    for source in SOURCES:
        r = results[source.key]
        print(f"[{source.key:6s}] {r['successes']}/{len(r['episodes'])} success, "
              f"mean pose error {np.mean([e['pose_error_mm'] for e in r['episodes']]):.1f} mm")
    print(f"wall {time.perf_counter() - t_start:.1f} s")
    if not args.no_video:
        print(f"wrote {args.out} and {args.log}")
    else:
        print(f"wrote {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
