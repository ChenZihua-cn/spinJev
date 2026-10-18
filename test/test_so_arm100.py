"""SO-ARM100 asset check: load the MJCF, verify joints/actuators/limits, step, render.

Run:      python test/test_so_arm100.py
Cluster-version pre-check (mujoco 3.3.4 is the ceiling there):
          .venv334/bin/python test/test_so_arm100.py --skip-render
One render backend per process (used internally by the backend probe):
          MUJOCO_GL=osmesa python test/test_so_arm100.py --render-only osmesa
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(os.path.dirname(HERE), "assets", "so_arm100")
MODEL_XML = os.path.join(ASSET_DIR, "so_arm100.xml")
SCENE_XML = os.path.join(ASSET_DIR, "scene.xml")
TABLE_XML = os.path.join(ASSET_DIR, "table_scene.xml")

EXPECTED_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper_joint",
]
EXPECTED_LINKS = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "jaw_link",
]

TEST_POSE = np.array([0.0, 0.6, -0.8, 0.0, 0.0, -0.22])

BACKENDS = ("glfw", "egl", "osmesa")


def load(mujoco, path):
    model = mujoco.MjModel.from_xml_path(path)
    print(
        f"[ok] loaded {os.path.relpath(path, os.path.dirname(HERE))} under mujoco {mujoco.__version__}: "
        f"nbody={model.nbody} njnt={model.njnt} nu={model.nu} ngeom={model.ngeom} nmesh={model.nmesh}"
    )
    return model


def joint_names(mujoco, model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def body_id(mujoco, model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise AssertionError(f"body {name!r} missing from the model")
    return bid


def report_joints(mujoco, model):
    print("\n     joint                type  range[rad]      axis          armature  fricloss")
    extra = [n for n in joint_names(mujoco, model) if n not in EXPECTED_JOINTS]
    if extra:
        raise AssertionError(f"unexpected joints: {extra}")
    ids = []
    for name in EXPECTED_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise AssertionError(f"joint {name!r} missing; found {joint_names(mujoco, model)}")
        ids.append(jid)
        lo, hi = model.jnt_range[jid]
        if not hi > lo:
            raise AssertionError(f"{name} is not limited (range {lo}..{hi})")
        ax = model.jnt_axis[jid]
        dof = model.jnt_dofadr[jid]
        print(
            f"     {name:<20} {mujoco.mjtJoint(model.jnt_type[jid]).name:<5} "
            f"[{lo:+.2f}, {hi:+.2f}] [{ax[0]:+.0f} {ax[1]:+.0f} {ax[2]:+.0f}] "
            f"  {model.dof_armature[dof]:.3f}     {model.dof_frictionloss[dof]:.3f}"
        )
    return ids


def report_actuators(mujoco, model, joint_ids):
    print("\n     actuator             kind     ctrlrange[rad]   forcerange[N*m]  kp")
    for name, jid in zip(EXPECTED_JOINTS, joint_ids):
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a < 0:
            raise AssertionError(f"no actuator named {name!r}")
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise AssertionError(f"{name} is not a joint actuator")
        if model.actuator_trnid[a, 0] != jid:
            raise AssertionError(f"{name} drives joint {model.actuator_trnid[a, 0]}, expected {jid}")
        if model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED:
            raise AssertionError(f"{name} has a non-fixed gain")
        lo, hi = model.actuator_ctrlrange[a]
        jlo, jhi = model.jnt_range[jid]
        if not np.allclose([lo, hi], [jlo, jhi], atol=1e-9):
            raise AssertionError(f"{name} ctrlrange {lo}..{hi} != joint range {jlo}..{jhi}")
        flo, fhi = model.actuator_forcerange[a]
        kp = model.actuator_gainprm[a, 0]
        print(
            f"     {name:<20} position [{lo:+.2f}, {hi:+.2f}]  [{flo:+.2f}, {fhi:+.2f}]   {kp:.1f}"
        )
    if model.nu != len(EXPECTED_JOINTS):
        raise AssertionError(f"expected {len(EXPECTED_JOINTS)} actuators, found {model.nu}")


def report_mass(mujoco, model):
    links = [body_id(mujoco, model, n) for n in EXPECTED_LINKS]
    shoulder = links[0]
    printed = float(model.body_mass[links].sum())
    total = float(model.body_subtreemass[shoulder])
    print(
        f"\n[ok] mass distal to the shoulder joint: {printed:.3f} kg printed links "
        f"+ {total - printed:.3f} kg servos = {total:.3f} kg"
    )
    for bid in links:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        welded = sum(model.body_mass[c] for c in range(model.nbody) if model.body_weldid[c] == bid)
        print(
            f"     {name:<16} own={model.body_mass[bid] + welded:.4f} kg  "
            f"diaginertia={np.array2string(model.body_inertia[bid], precision=6)}"
        )
    if not 0.8 < total < 1.0:
        raise AssertionError(
            f"moving mass {total:.3f} kg: an assembled SO-ARM100 carries 5x55 g of STS3215 "
            f"beyond the shoulder joint, on top of ~0.61 kg of printed parts"
        )


def check_physics(mujoco, model, steps):
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    if not np.all((TEST_POSE >= lo) & (TEST_POSE <= hi)):
        raise AssertionError(f"TEST_POSE {TEST_POSE} is outside the joint ranges")
    data.ctrl[:] = TEST_POSE
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - t0

    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise AssertionError("qpos/qvel contain NaN/Inf after stepping")
    err = float(np.max(np.abs(data.qpos - TEST_POSE)))
    if err > 0.05:
        raise AssertionError(f"position actuators did not reach the pose (max error {err:.3f} rad)")
    viol = max(float(np.max(lo - data.qpos)), float(np.max(data.qpos - hi)), 0.0)
    print(
        f"[ok] physics: {steps} steps in {elapsed:.3f}s ({steps / elapsed / 1000:.1f}k steps/s), "
        f"max |q - q_target| = {err:.4f} rad, limit overshoot {viol:.1e} rad"
    )
    return data


def jaw_geom_id(mujoco, model):
    """The collision geom of the moving jaw (last geom attached to jaw_link)."""
    jaw = body_id(mujoco, model, "jaw_link")
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == jaw]
    if not geoms:
        raise AssertionError("jaw_link has no geoms")
    return geoms[-1]


def jaw_frame(mujoco, model, gid, jid, qvalue):
    """(jaw body id, data) for the arm at rest with the gripper at qvalue."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[jid]] = qvalue
    mujoco.mj_forward(model, data)
    return model.geom_bodyid[gid], data


def world_of(jaw, data, local_point):
    rot = data.xmat[jaw].reshape(3, 3).T
    return data.xpos[jaw] + np.asarray(local_point) @ rot


def jaw_corner(mujoco, model, gid, jid):
    """Corner of the moving jaw's bounding box farthest from the gripper pivot axis."""
    aabb = np.asarray(model.geom_aabb).reshape(-1)
    centre, half = aabb[gid * 6 : gid * 6 + 3], aabb[gid * 6 + 3 : gid * 6 + 6]
    corners = np.array(
        [centre + np.array([sx, sy, sz]) * half for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    jaw, data = jaw_frame(mujoco, model, gid, jid, model.jnt_range[jid][0])
    anchor, axis = data.xanchor[jid], data.xaxis[jid]
    relative = world_of(jaw, data, corners) - anchor
    perpendicular = np.linalg.norm(relative - (relative @ axis)[:, None] * axis, axis=1)
    return corners[np.argmax(perpendicular)]


def check_gripper(mujoco, model):
    gid = jaw_geom_id(mujoco, model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
    lo, hi = model.jnt_range[jid]
    corner = jaw_corner(mujoco, model, gid, jid)
    jaw_lo, data_lo = jaw_frame(mujoco, model, gid, jid, lo)
    jaw_hi, data_hi = jaw_frame(mujoco, model, gid, jid, hi)
    chord = float(np.linalg.norm(world_of(jaw_hi, data_hi, corner) - world_of(jaw_lo, data_lo, corner)))
    print(f"[ok] gripper: {lo:+.2f} -> {hi:+.2f} rad swings the jaw corner {chord * 1000:.1f} mm")
    if chord < 0.02:
        raise AssertionError(f"jaw corner only travels {chord * 1000:.2f} mm; gripper limits look wrong")


def check_workspace(mujoco, model, samples=512, seed=0):
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    base = body_id(mujoco, model, "base")
    wrist = body_id(mujoco, model, "wrist_2_link")
    radii, heights = [], []
    for q in rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1], size=(samples, model.nq)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        offset = data.xpos[wrist] - data.xpos[base]
        radii.append(float(np.hypot(offset[0], offset[1])))
        heights.append(float(offset[2]))
    r, h = np.array(radii), np.array(heights)
    p5, p50, p95 = np.percentile(r, [5, 50, 95])
    print(
        f"[ok] workspace (wrist_2 relative to base, {samples} uniform joint samples): "
        f"horizontal reach p5/p50/p95 = {p5:.3f}/{p50:.3f}/{p95:.3f} m, "
        f"height {h.min():.3f}..{h.max():.3f} m"
    )


def check_task_scene(mujoco, steps=4000):
    model = load(mujoco, TABLE_XML)
    if model.nmocap != 1:
        raise AssertionError(f"expected 1 mocap body (the occluder), found {model.nmocap}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead") < 0:
        raise AssertionError("no overhead camera in the task scene")

    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target")]
    start = data.qpos[adr : adr + 2].copy()
    # Start in the pose rather than sweeping into it: on the way there the arm would plough
    # through the target, which is a fact about the pose sequence, not about the scene.
    data.qpos[:6] = TEST_POSE
    data.ctrl[:] = TEST_POSE
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        raise AssertionError("state went non-finite in the task scene")

    pos = data.qpos[adr : adr + 3]
    half_z = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")][2])
    if abs(pos[2] - half_z) > 0.003:
        raise AssertionError(f"target did not settle on the table top (z={pos[2]:.4f}, half-size {half_z})")
    drift = float(np.linalg.norm(pos[:2] - start))
    if drift > 0.005:
        raise AssertionError(f"target slid {drift * 1000:.1f} mm while the arm held a pose")
    print(f"[ok] task scene: target rests at z={pos[2]:.4f} m, {drift * 1000:.1f} mm drift in {steps} steps")
    check_reach(mujoco, model, pos[:2])
    return model


def check_reach(mujoco, model, target_xy, samples=4000, seed=0, tol=0.03):
    """At least one legal joint config must bring the jaw to the target, or the task is not feasible."""
    gid = jaw_geom_id(mujoco, model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
    corner = jaw_corner(mujoco, model, gid, jid)
    jaw = model.geom_bodyid[gid]
    block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    centre = np.array([target_xy[0], target_xy[1], float(model.geom_size[block][2])])
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    best = np.inf
    for q in rng.uniform(model.jnt_range[:6, 0], model.jnt_range[:6, 1], size=(samples, 6)):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        best = min(best, float(np.linalg.norm(world_of(jaw, data, corner) - centre)))
    print(f"[ok] reachability: jaw corner gets within {best * 1000:.1f} mm of the target ({samples} configs)")
    if best > tol:
        raise AssertionError(f"target out of reach: nearest approach {best * 1000:.1f} mm > {tol * 1000:.0f} mm")


def save_ppm(img, path):
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def render_frame(mujoco, model, outdir, tag, camera=None, park_occluder=False):
    data = mujoco.MjData(model)
    data.ctrl[:] = TEST_POSE
    if model.nmocap and park_occluder:
        data.mocap_pos[0] = [0.5, 0.5, 0.5]
    for _ in range(600):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=360, width=480)
    try:
        if camera is None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.0, 0.0, 0.12]
            cam.distance, cam.azimuth, cam.elevation = 0.75, 45, -20
        else:
            cam = camera
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
    finally:
        close = getattr(renderer, "close", None)
        if close:
            close()
    if img.shape != (360, 480, 3):
        raise AssertionError(f"unexpected image shape {img.shape}")
    if float(img.std()) < 1.0:
        raise AssertionError(f"rendered image is blank (std={img.std():.2f})")
    out = os.path.join(outdir, f"render_so100_{tag}.ppm")
    save_ppm(img, out)
    print(f"       mean={img.mean():.1f} std={img.std():.1f} -> {out}")
    return img


def check_occlusion(mujoco, model, outdir, tag):
    """The overhead camera looks straight down at the target, so the target is the image centre."""
    clear = render_frame(mujoco, model, outdir, f"table_clear_{tag}", camera="overhead", park_occluder=True)
    blocked = render_frame(mujoco, model, outdir, f"table_blocked_{tag}", camera="overhead")
    h, w, _ = clear.shape
    patch = (slice(h // 2 - 30, h // 2 + 30), slice(w // 2 - 30, w // 2 + 30))
    drop = float(clear[patch].mean() - blocked[patch].mean())
    whole = float(np.abs(clear.astype(np.int32) - blocked.astype(np.int32)).mean())
    print(f"[ok] occluder darkens the target patch by {drop:.0f} grey levels ({whole:.1f} over the whole frame)")
    if drop < 50.0:
        raise AssertionError(f"occluder does not hide the target (patch only darkened by {drop:.0f} levels)")


def render_all(mujoco, outdir, tag):
    render_frame(mujoco, load(mujoco, SCENE_XML), outdir, tag)
    table = load(mujoco, TABLE_XML)
    check_occlusion(mujoco, table, outdir, tag)


def check_backend(mujoco, outdir, backend):
    env = dict(os.environ)
    if backend is not None:
        env["MUJOCO_GL"] = backend
    else:
        env.pop("MUJOCO_GL", None)
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--render-only",
        backend or "auto",
        "--outdir",
        outdir,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    label = backend or "auto"
    if proc.returncode == 0:
        print(f"[ok] render backend {label}")
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip().startswith(("mean=", "[ok] occluder")):
                print(f"       {line.strip()}")
        return label
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    print(f"[fail] render backend {label}: {tail[-1] if tail else 'no output'}")
    return None


def check_rendering(mujoco, outdir):
    probes = [os.environ.get("MUJOCO_GL")] if os.environ.get("MUJOCO_GL") else [None]
    probes += [b for b in BACKENDS if b not in probes]
    return [label for label in (check_backend(mujoco, outdir, b) for b in probes) if label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--render-only", help="internal: render one frame with the ambient MUJOCO_GL and exit")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        print(f"[FAIL] cannot import mujoco: {exc}")
        return 1

    for path in (MODEL_XML, SCENE_XML, TABLE_XML):
        if not os.path.exists(path):
            print(f"[FAIL] missing asset {path}")
            return 1

    os.makedirs(args.outdir, exist_ok=True)
    model = load(mujoco, MODEL_XML)
    joint_ids = report_joints(mujoco, model)
    report_actuators(mujoco, model, joint_ids)
    report_mass(mujoco, model)
    check_gripper(mujoco, model)
    check_workspace(mujoco, model)
    check_physics(mujoco, model, args.steps)
    check_task_scene(mujoco, args.steps)

    if args.render_only:
        render_all(mujoco, args.outdir, args.render_only)
        return 0
    if args.skip_render:
        print("[skip] rendering (--skip-render)")
        print("\n[PASS] SO-ARM100 asset loads, steps and reports sane kinematics.")
        return 0

    working = check_rendering(mujoco, args.outdir)
    if not working:
        print("\n[FAIL] asset is physically fine but no offscreen render backend is usable.")
        return 1
    print(f"\n[PASS] SO-ARM100 MJCF verified. Working render backends: {', '.join(working)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
