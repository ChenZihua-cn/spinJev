"""阶段 1a acceptance: the spine's primitives, guards and pre-filter, plus the calibration
numbers the task layer is built on.

Run:      .venv/bin/python test/test_spine.py
Cluster-version pre-check (mujoco 3.3.4 is the ceiling there):
          .venv334/bin/python test/test_spine.py
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from spinjev.backends.sim.mujoco_sim import TABLE_SCENE, Sim  # noqa: E402
from spinjev.bus.schema import TAU_LIMIT  # noqa: E402
from spinjev.spine.guard import FATAL, RADIUS_FENCE, TABLE_FORCE_LIMIT, Guard  # noqa: E402
from spinjev.spine.primitives import CLOSE_Q, OPEN_Q, PHYS_DT, REACHED_TOL, Primitive, Spine  # noqa: E402
from spinjev.perception.simulator import Perceptor  # noqa: E402
from spinjev.tasks.pick_place import (  # noqa: E402
    DROP_ZONE,
    R_NEAR,
    R_REACH,
    THETA_MAX,
    PickPlace,
    TaskSpec,
)

np.set_printoptions(precision=3, suppress=True)
OBJECT_XY = (0.0, -0.24)


def quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def points_in_wrist_frame(sim, geom):
    """Real material points of a mesh geom, expressed in the wrist_2 frame. The AABB is not good
    enough here: it is a box around a curved finger, and its corners are empty space."""
    m, d = sim.model, sim.data
    did = m.geom_dataid[geom]
    verts = m.mesh_vert[m.mesh_vertadr[did] : m.mesh_vertadr[did] + m.mesh_vertnum[did]]
    body = m.geom_bodyid[geom]
    in_body = verts @ quat_mat(m.geom_quat[geom]).T + m.geom_pos[geom]
    world = in_body @ quat_mat(d.xquat[body]).T + d.xpos[body]
    return (world - d.xpos[sim.ee_body]) @ quat_mat(d.xquat[sim.ee_body])


def describe_gripper(sim):
    """Where along the approach axis do the two jaws actually have material, and how far apart
    are they there? This decides what object heights this arm can grip from a flat table."""
    fixed, moving = jaw_geoms(sim)
    print("\n     jaw profile in the wrist frame (-y is out along the approach axis, x is the")
    print("     closing direction): how much of each jaw exists at that depth, and the gap")
    print("     depth   fixed x-extent        moving x-extent       surface gap")
    for q in (CLOSE_Q, 0.6, OPEN_Q):
        sim.data.qpos[sim.gadr] = q
        mujoco.mj_forward(sim.model, sim.data)
        pts = {k: points_in_wrist_frame(sim, g) for k, g in (("f", fixed), ("m", moving))}
        print(f"     gripper q={q:+.2f} rad")
        for depth in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            row = []
            for key in ("f", "m"):
                p = pts[key]
                band = p[(p[:, 1] > -(depth + 0.012)) & (p[:, 1] < -(depth - 0.012))]
                row.append((band[:, 0].min(), band[:, 0].max()) if len(band) > 3 else None)
            if row[0] is None or row[1] is None:
                print(f"      {depth * 1000:5.0f} mm   no material on one jaw")
                continue
            (f_lo, f_hi), (m_lo, m_hi) = row
            gap = min(abs(f_hi - m_lo), abs(m_hi - f_lo))
            print(f"      {depth * 1000:5.0f} mm   [{f_lo * 1000:+5.1f},{f_hi * 1000:+5.1f}] mm   "
                  f"[{m_lo * 1000:+5.1f},{m_hi * 1000:+5.1f}] mm   {gap * 1000:5.1f} mm")


def closing_direction(sim):
    """Which way the jaws close when the approach axis points down, by waypoint. The block rests
    axis-aligned, so a closing heading near 0 or 180 deg straddles its faces; near 90 deg a finger
    drives into the top face and pushes the block away instead of gripping it."""
    sim.reset(OBJECT_XY)
    print("\n     closing heading of the gripper-down solution, by waypoint:")
    for th in (0.0, 15.0, 30.0, 45.0):
        for r in (0.24, 0.30):
            goal = np.array([r * np.sin(np.radians(th)), -r * np.cos(np.radians(th)), 0.13])
            q, err, axis, close = sim.kin.solve(goal, sim.kin.qref, iters=200)
            print(f"      theta={th:+5.1f} r={r:.2f}: residual {err * 1000:5.1f} mm "
                  f"axis_z={axis[2]:+.3f} roll={q[4]:+.2f} rad "
                  f"-> closing heading {sim.kin.closing_heading(close):+6.1f} deg")


def envelope(sim, thetas=(0.0, 20.0, -20.0), rs=np.arange(0.16, 0.42, 0.02), zs=np.arange(0.06, 0.24, 0.02)):
    """The reach envelope as the pre-filter sees it: '#' IK + pose + limits ok *and* the servos
    can hold the configuration against gravity, 'x' reachable but torque-infeasible, '.' a miss.
    R_* and the carry heights in TaskSpec come from this grid."""
    print("\n     gripper-down envelope, torque-checked ('#' legal, 'x' too heavy, '.' miss)")
    for th in thetas:
        print(f"     theta={th:+.0f} deg")
        for z in zs:
            row = []
            for r in rs:
                goal = np.array([r * np.sin(np.radians(th)), -r * np.cos(np.radians(th)), z])
                q, err, axis, _ = sim.kin.solve(goal, sim.kin.qref, iters=120)
                torque = float(np.max(np.abs(sim.static_torque(np.concatenate([q, [OPEN_Q]]))[:5])))
                if err >= 0.010 or axis[2] >= -0.90:
                    row.append("." if err < 0.030 else " ")
                else:
                    row.append("#" if torque < 0.90 * TAU_LIMIT else "x")
            print(f"      z={z:.2f} " + "".join(row))
        print("           " + "".join(f"{r:.2f}"[:3] + " " for r in rs[:4]))


def success_rate(sim, spec, n=20, seed=5, perceptor=None):
    """Run the scripted pick at N random reachable placements and report how many complete.
    This is the number that decides whether the task geometry is usable, so it is measured over
    the distribution the experts will actually sample, not at one lucky waypoint."""
    perc = perceptor if perceptor is not None else oracle(seed)
    env = PickPlace(sim=sim, perceptor=perc, spec=spec, seed=seed)
    rng = np.random.default_rng(seed)
    fails = {}
    done = 0
    for i in range(n):
        r, th = rng.uniform(R_NEAR, R_REACH), rng.uniform(-THETA_MAX, THETA_MAX)
        xy = np.array([r * math.sin(th), -r * math.cos(th)])
        env.reset(xy=xy, unreachable=False, occluded=False)
        # env.reset() re-seeds the perceptor to its original seed; override it so each placement
        # gets its own detection noise instead of 20 copies of one draw.
        perc.reset(seed=seed * 1000003 + i)
        scripted_pick(env)
        if env.out.success:
            done += 1
        else:
            key = env.out.reason or ("held" if env.out.held_ticks else "never gripped")
            fails[key] = fails.get(key, 0) + 1
    print(f"[rate] scripted pick succeeded {done}/{n} at reachable placements "
          f"(r {R_NEAR:.2f}..{R_REACH:.2f} m, |theta| {math.degrees(THETA_MAX):.0f} deg)")
    for reason, count in sorted(fails.items(), key=lambda kv: -kv[1]):
        print(f"       {count:3d} x {reason}")
    return done, n


def oracle(seed=0):
    """A noiseless perceptor behind the *same* interface. Stage 1a asks whether the primitives can
    complete a move, so it must not be graded through a pose estimator; the data path uses the noisy
    one and --rate-noisy measures that separately."""
    return Perceptor(sigma_clear=0.0, sigma_occluded=0.0, miss_rate=0.0, seed=seed)


def load_sim(xml=None):
    sim = Sim(xml=xml or TABLE_SCENE, object_xy=OBJECT_XY)
    if abs(sim.dt - PHYS_DT) > 1e-12:
        raise AssertionError(f"model timestep {sim.dt} != spine PHYS_DT {PHYS_DT}")
    print(f"[ok] sim ready: physics {sim.dt * 1000:.0f} ms -> {1 / sim.dt:.0f} Hz servo, "
          f"{1 / (sim.dt * Spine(sim).decim):.0f} Hz decisions")
    return sim


def jaw_geoms(sim):
    """The two colliding jaw geoms: one welded to wrist_2, one on the moving jaw."""
    out = []
    for body in ("wrist_2_link", "jaw_link"):
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body)
        geoms = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid and sim.model.geom_contype[g] > 0]
        if not geoms:
            raise AssertionError(f"{body} has no collision geom")
        out.append(geoms[-1])
    return out


def check_gripper_end(sim):
    """The spine assumes the low end of the gripper range is jaws-shut. Re-measure, don't trust."""
    fixed, moving = jaw_geoms(sim)
    gap = []
    for q in (CLOSE_Q, OPEN_Q):
        sim.data.qpos[sim.gadr] = q
        mujoco.mj_forward(sim.model, sim.data)
        gap.append(float(np.linalg.norm(sim.data.geom_xpos[fixed] - sim.data.geom_xpos[moving])))
    if gap[1] < gap[0] + 0.01:
        raise AssertionError(f"the gripper range is not the way the spine assumes: {gap}")
    print(f"[ok] gripper: {CLOSE_Q:+.2f} rad = {gap[0] * 1000:.0f} mm jaw separation (shut), "
          f"{OPEN_Q:+.2f} rad = {gap[1] * 1000:.0f} mm (open)")


def check_ik(sim, n=60, seed=0):
    """Goals drawn from inside the envelope the task layer actually uses, at the tolerance the
    spine itself arrives at (10 mm), not a tighter one nothing can hit."""
    rng = np.random.default_rng(seed)
    worst, misses, total = 0.0, [], 0.0
    for _ in range(n):
        r, th, z = rng.uniform(0.22, 0.32), rng.uniform(-0.35, 0.35), rng.uniform(0.06, 0.18)
        goal = np.array([r * np.sin(th), -r * np.cos(th), z])
        t0 = time.perf_counter()
        q, err, axis, close = sim.kin.solve(goal, sim.kin.qref, iters=120)
        total += time.perf_counter() - t0
        worst = max(worst, err)
        if err > REACHED_TOL or axis[2] > -0.90:
            misses.append((np.round(goal, 3), round(err, 4), round(float(axis[2]), 3)))
    print(f"[ok] IK: worst residual {worst * 1000:.1f} mm over {n} goals in the working envelope, "
          f"{len(misses)} misses, mean solve {total / n * 1000:.1f} ms")
    for miss in misses[:5]:
        print(f"       miss {miss}")
    if len(misses) > 0.1 * n:
        raise AssertionError(f"too many IK misses: {len(misses)}/{n}")


DEPTHS = (0.085, 0.095, 0.105, 0.115)
LATERALS = (-0.055, -0.045, -0.035, -0.025, -0.015)
SLIP_MAX = 0.05
AIM_TICKS = 20      # detections averaged before descending (2026-10-27, independent noise
                    # per placement: 12/20 at 1, 17/20 at 20, 19/20 at 60; oracle 17/20)


def calibrate_grasp(sim):
    """Two-dimensional descent calibration: how far below the grasp point the wrist sits, and how
    far to the side (against the jaw channel offset) it stands. Grip success is read off the
    physics -- the block either comes up with the hand or it does not."""
    print("\n     descent sweep, block at (0, -0.24) standing on the table:")
    print("       depth  lateral   approach     lifted to   slip    verdict")
    gripping = []
    for depth in DEPTHS:
        for lat in LATERALS:
            spec = TaskSpec(grasp_depth=depth, grasp_lateral=lat)
            env = PickPlace(sim=sim, perceptor=oracle(7), spec=spec, seed=7)
            env.reset(xy=OBJECT_XY, unreachable=False, occluded=False)
            th = env.heading_of(OBJECT_XY)
            status = env.drive(Primitive.APPROACH, env.wrist_goal(OBJECT_XY, env.block_z, th))
            if status != "reached":
                print(f"       {depth:.3f}  {lat:+.3f}   {status:<11} --  (guard {env.out.tripped})")
                continue
            env.drive(Primitive.CLOSE)
            env.drive(Primitive.LIFT, [0.0, 0.0, spec.lift_height])
            obj = sim.truth().obj_pos
            lifted = float(obj[2])
            slip = float(np.linalg.norm(obj[:2] - np.array(OBJECT_XY)))
            ok = lifted > env.hold_z and slip < SLIP_MAX
            if ok:
                gripping.append((depth, lat))
            print(f"       {depth:.3f}  {lat:+.3f}   reached     {lifted:.4f}   {slip * 1000:4.0f} mm  "
                  f"{'GRIP' if ok else 'no grip'}")
    print(f"     gripping pairs: {gripping}")
    if not gripping:
        raise AssertionError("no (depth, lateral) pair grips the block -- the task is not achievable")
    return gripping


def check_guards(sim, spec):
    guard = Guard(sim.qlo, sim.qhi, TAU_LIMIT)
    inboard = np.array([0.2, -0.2, 0.1])
    st = guard.check(np.zeros(6), np.zeros(6), np.array([RADIUS_FENCE + 0.2, 0.0, 0.1]))
    if "out_of_workspace" not in st.trips or not st.fatal:
        raise AssertionError(f"workspace fence did not trip: {st}")
    st = guard.check(np.zeros(6), np.zeros(6), inboard, table_force=TABLE_FORCE_LIMIT + 1)
    if not st.fatal or "table_hit" not in st.trips:
        raise AssertionError("pressing the table did not trip the guard")
    st = guard.check(np.zeros(6), np.zeros(6), inboard, table_force=0.5)
    if st.fatal or not st.touching:
        raise AssertionError(f"a fingertip graze must be reported, not fatal: {st}")
    st = guard.check(np.zeros(6), np.full(6, TAU_LIMIT), inboard)
    if st.fatal:
        raise AssertionError("one saturated tick must not be fatal")
    for _ in range(30):
        st = guard.check(np.zeros(6), np.full(6, TAU_LIMIT), inboard)
    if "arm_straining" not in st.trips:
        raise AssertionError("sustained saturation did not trip")

    env = PickPlace(sim=sim, perceptor=oracle(7), spec=spec, seed=7)
    env.reset(xy=OBJECT_XY, unreachable=False, occluded=False)
    status = env.drive(Primitive.APPROACH, np.array([OBJECT_XY[0], OBJECT_XY[1], 0.01]))
    frozen = sim.data.qpos[sim.arm_qadr].copy()
    sim.step(500)
    drift = float(np.max(np.abs(sim.data.qpos[sim.arm_qadr] - frozen)))
    if "table_hit" not in status or drift > 0.05:
        raise AssertionError(f"driving into the table: status={status}, drift after trip={drift:.3f} rad")
    print(f"[ok] guards: {FATAL} all trip, a graze only reports, and a trip freezes the "
          f"setpoints (drift {drift:.4f} rad)")


def check_prefilter(sim, spec):
    env = PickPlace(sim=sim, perceptor=oracle(7), spec=spec, seed=7)
    env.reset(xy=OBJECT_XY, unreachable=False, occluded=False)
    legal = env.candidates()
    if Primitive.APPROACH not in legal or Primitive.CLOSE not in legal:
        raise AssertionError(f"default block should allow approach+close: {legal.reasons}")
    if Primitive.LIFT in legal:
        raise AssertionError("nothing is gripped yet, so lift should be illegal")

    env.reset(xy=(0.0, -0.62), unreachable=True, occluded=False)
    legal = env.candidates()
    if "approach:grasp" not in legal.reasons or legal.reasons["approach:grasp"] == "ok":
        raise AssertionError(f"a block 0.62 m out should fail the grasp waypoint: {legal.reasons}")
    if Primitive.APPROACH not in legal:
        raise AssertionError("drop and home are still reachable, so APPROACH itself stays legal")
    near = env.reset(xy=(0.0, -0.12), unreachable=True, occluded=False)
    legal_in = env.candidates()
    if legal_in.reasons.get("approach:grasp", "ok") == "ok":
        raise AssertionError(f"a block folded too close should fail the grasp waypoint: {legal_in.reasons}")
    print(f"[ok] pre-filter: far away -> grasp:{legal.reasons['approach:grasp']}; too close -> "
          f"grasp:{legal_in.reasons['approach:grasp']}; grip gate -> {legal.reasons['lift']} "
          f"(near={near.radius:.2f})")

    env.reset(xy=OBJECT_XY, unreachable=False, occluded=False)
    sim.data.qpos[sim.gadr] = CLOSE_Q
    if Primitive.CLOSE in env.candidates():
        raise AssertionError("an already-shut gripper should not be offered CLOSE")
    sim.data.qpos[sim.gadr] = float(sim.qhi[-1])
    if Primitive.RELEASE in env.candidates():
        raise AssertionError("a fully-open gripper should not be offered RELEASE")
    sim.data.qpos[sim.gadr] = OPEN_Q


def scripted_pick(env):
    """The full sequence: home -> hover -> descend -> close -> lift -> carry -> release."""
    spec = env.spec
    p = env.aim(AIM_TICKS)  # 对准: the grasp tolerance is smaller than one detection's error
    th = env.heading_of(p.pos[:2])
    for prim, goal in (
        (Primitive.APPROACH, spec.home),
        (Primitive.APPROACH, env.wrist_goal(p.pos[:2], p.pos[2], th, lift=spec.pregrasp_clear)),
        (Primitive.APPROACH, env.wrist_goal(p.pos[:2], p.pos[2], th)),
        (Primitive.CLOSE, None),
        (Primitive.LIFT, [0.0, 0.0, spec.lift_height]),
        (Primitive.APPROACH, [DROP_ZONE[0], DROP_ZONE[1], spec.lift_height]),
        (Primitive.RELEASE, None),
    ):
        status = env.drive(prim, goal)
        if env.out.done:
            return status
        if status not in ("reached", "held", "blocked"):
            if goal is None:
                print(f"       {prim.value:<9} -> {status}")
            else:
                here = env.sim.kin.ee()[0]
                print(f"       {prim.value:<9} -> {status}: goal {np.round(np.asarray(goal), 3)} "
                      f"stopped at {np.round(here, 3)}, err {np.linalg.norm(here - goal) * 1000:.0f} mm")
    env.wait(120)
    return env.spine.status


def check_full_pick(sim, spec, need=2):
    """The acceptance criterion for 阶段 1a: the primitive set, driven by a fixed sequence, can
    complete a pick-and-place. It has to work at more than one placement, but it is not required
    to work at all of them -- the measured grip basin is ~10 mm wide, so some of these placements
    legitimately fail, and the failure is data, not a bug (see the note on TaskSpec)."""
    ok_count = 0
    for xy in [(0.0, -0.24), (0.10, -0.28), (-0.12, -0.28), (0.05, -0.30)]:
        env = PickPlace(sim=sim, perceptor=oracle(11), spec=spec, seed=11)
        env.reset(xy=xy, unreachable=False, occluded=False)
        t0 = time.perf_counter()
        scripted_pick(env)
        wall = time.perf_counter() - t0
        out = env.out
        ok_count += bool(out.success)
        print(
            f"[{'ok' if out.success else 'fail'}] pick at ({xy[0]:+.2f}, {xy[1]:+.2f}) "
            f"r={np.linalg.norm(xy):.2f}: {out.reason or 'placed'}, reward {out.reward:+.2f}, "
            f"{out.ticks} ticks, {wall:.1f} s wall"
        )
    if ok_count < need:
        raise AssertionError(f"only {ok_count}/{need} scripted picks completed")
    print(f"[ok] scripted pick-and-place completes at {ok_count}/4 placements")


def throughput(sim):
    spine = Spine(sim)
    sim.reset(OBJECT_XY)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        spine.tick()
        sim.step(spine.decim)
    wall = time.perf_counter() - t0
    print(
        f"[ok] throughput: {n} decision ticks = {n * spine.decim * sim.dt:.0f} s of physics in "
        f"{wall:.2f} s -> {n / wall:.0f} ticks/s, {1000 * wall / n:.2f} ms/tick "
        f"(idle-branch solve {1000 * spine.solve_s:.2f} ms)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve-only", action="store_true", help="just re-run the descent calibration")
    ap.add_argument("--profile", action="store_true", help="just print the jaw profile and exit")
    ap.add_argument("--envelope", action="store_true", help="just print the reach envelope grid")
    ap.add_argument("--rate", type=int, help="calibrate, then run N scripted picks and report the rate")
    ap.add_argument("--rate-noisy", action="store_true", help="run --rate through the real noisy detector")
    ap.add_argument("--xml", help="scene variant to measure, for target-shape experiments")
    args = ap.parse_args()

    sim = load_sim(args.xml)
    check_gripper_end(sim)
    if args.profile:
        describe_gripper(sim)
        closing_direction(sim)
        return 0
    if args.envelope:
        closing_direction(sim)
        envelope(sim)
        return 0
    check_ik(sim)
    gripping = calibrate_grasp(sim)
    depth, lateral = gripping[len(gripping) // 2]
    spec = TaskSpec(grasp_depth=depth, grasp_lateral=lateral,
                    min_wrist_z=sim.block_half_z + min(d for d, _ in gripping))
    print(f"\n[ok] calibrated: grasp_depth={depth:.3f} grasp_lateral={lateral:+.3f} "
          f"min_wrist_z={spec.min_wrist_z:.3f} (from {gripping})")
    if args.solve_only:
        return 0
    check_guards(sim, spec)
    check_prefilter(sim, spec)
    if args.rate:
        perc = Perceptor(seed=3) if args.rate_noisy else oracle(3)
        label = "noisy detector (sigma 10 mm / 100 mm occluded)" if args.rate_noisy else "oracle pose"
        print(f"     perception channel: {label}")
        success_rate(sim, spec, n=args.rate, perceptor=perc)
        return 0
    check_full_pick(sim, spec)
    throughput(sim)
    print("\n[PASS] spine primitives complete a scripted pick-and-place.")
    print(f"       measured constants for TaskSpec: grasp_depth={depth:.3f} "
          f"grasp_lateral={lateral:+.3f} min_wrist_z={spec.min_wrist_z:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
