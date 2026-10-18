"""阶段 1b acceptance: perception simulator, ring buffer, feature compressor, decision log.

No MuJoCo and no assets: every frame here is synthesized, which is the point -- the bus layer has
to be checkable in a second and must not be able to reach the simulator even by import. The second
interpreter below is the version-parity gate the project requires.

Run:      .venv/bin/python test/test_bus.py
Cluster-version pre-check (mujoco 3.3.4 is the ceiling there):
          .venv334/bin/python test/test_bus.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from spinjev.bus.compressor import SLIP_LIMIT, SLIP_TICKS, Compressor  # noqa: E402
from spinjev.bus.eventlog import SCHEMA_VERSION, Decision, EventLog, read  # noqa: E402
from spinjev.bus.ringbuffer import TICK_DT, WINDOW_S, WINDOW_TICKS, RingBuffer  # noqa: E402
from spinjev.bus.schema import (  # noqa: E402
    NJNT,
    OBS_DIM,
    OBS_LAYOUT,
    Percept,
    Proprio,
    Truth,
    obs_vector,
)
from spinjev.perception import Perceptor  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

# Joint travel of this asset, written out instead of read from the model so this file never needs
# MuJoCo. Keep in step with the jnt_range values in assets/so_arm100/so_arm100.xml.
QLO = np.array([-2.0, -1.75, -1.75, -3.5, -1.0, -0.22])
QHI = np.array([2.0, 1.75, 1.57, 0.2, 4.0, 2.0])
OBJ = np.array([0.0, -0.24, 0.03])
GRASP = OBJ - np.array([0.0, 0.09, 0.0])  # the block sits 90 mm out along the approach axis
TRUTH = Truth(obj_pos=OBJ.copy(), obj_vel=np.array([0.0, 0.0, 0.01]), ncontact=3)


def span(name):
    names = [n for n, _ in OBS_LAYOUT]
    i = names.index(name)
    lo = sum(d for _, d in OBS_LAYOUT[:i])
    return lo, lo + OBS_LAYOUT[i][1]


def section(vec, name):
    lo, hi = span(name)
    return vec[lo:hi]


def clean_perceptor(seed=0):
    return Perceptor(seed=seed, sigma_clear=0.0, sigma_occluded=0.0, miss_rate=0.0, conf_jitter=0.0)


def run_ticks(perceptor, comp, ring, steps, grasp_of, obj_of, occluded=False):
    """Push, compress -- the order a collection loop uses: the frame reaches the window before the
    compressor reads it. Returns one vector per tick."""
    vectors = []
    for k in range(steps):
        prop = Proprio(t=k * TICK_DT, grasp_pos=np.asarray(grasp_of(k), dtype=np.float64))
        ring.push(prop)
        perc = perceptor(Truth(obj_pos=np.asarray(obj_of(k), dtype=np.float64)), occluded=occluded)
        vectors.append(comp(ring, prop, perc))
    return vectors


def check_layout():
    names = [n for n, _ in OBS_LAYOUT]
    assert len(names) == len(set(names)), f"duplicate section names: {names}"
    assert OBS_DIM == sum(d for _, d in OBS_LAYOUT)
    assert ("joint_margin", NJNT) in OBS_LAYOUT and ("grip_slip", 1) in OBS_LAYOUT

    probe = {name: np.full(dim, i) for i, (name, dim) in enumerate(OBS_LAYOUT)}
    vec = obs_vector(probe)
    assert vec.shape == (OBS_DIM,), f"obs_vector gave {vec.shape}, expected ({OBS_DIM},)"
    for i, (name, _) in enumerate(OBS_LAYOUT):
        lo, hi = span(name)
        assert np.all(vec[lo:hi] == i), f"section {name} is not at offset {lo}"
    print(f"[ok] OBS_LAYOUT: {len(names)} sections, {OBS_DIM} floats, in order {names}")


def check_ring():
    assert abs(TICK_DT - 0.02) < 1e-12, f"the spine ticks every {TICK_DT} s, not 20 ms"
    assert WINDOW_TICKS == 100 == int(round(WINDOW_S / TICK_DT)), WINDOW_TICKS

    ring = RingBuffer()
    assert len(ring) == 0 and ring.latest() is None and ring.oldest() is None
    assert ring.span() == 0.0 and not ring.full()
    assert abs(ring.window_s - WINDOW_S) < 1e-12 and ring.seconds_per_frame == TICK_DT

    n = WINDOW_TICKS + 150
    for i in range(n):
        ring.push(Proprio(t=i * TICK_DT))
    assert len(ring) == ring.capacity == WINDOW_TICKS, len(ring)
    assert ring.full() and ring.dropped == n - WINDOW_TICKS, ring.dropped
    assert ring.oldest().t == (n - WINDOW_TICKS) * TICK_DT, ring.oldest().t
    assert ring.latest().t == (n - 1) * TICK_DT
    assert abs(ring.span() - (WINDOW_TICKS - 1) * TICK_DT) < 1e-12, ring.span()
    assert ring.tail(3)[-1] is ring.latest() and len(ring.tail(3)) == 3
    assert [f.t for f in ring.frames()] == [i * TICK_DT for i in range(n - WINDOW_TICKS, n)]
    span_held = ring.span()
    ring.clear()
    assert len(ring) == 0 and ring.dropped == 0 and not ring.full()
    print(f"[ok] ring buffer: {WINDOW_TICKS} frames x {TICK_DT * 1000:.0f} ms = "
          f"{WINDOW_S:.1f} s window; {n} pushes evicted {n - WINDOW_TICKS} and held "
          f"{span_held:.2f} s of history")


def check_reproducible():
    a, b = Perceptor(seed=7), Perceptor(seed=7)
    for k in range(60):
        occluded = bool(k % 2)
        pa, pb = a(TRUTH, occluded=occluded), b(TRUTH, occluded=occluded)
        assert np.array_equal(pa.pos, pb.pos), "same seed did not reproduce the pose stream"
        assert pa.valid == pb.valid and pa.confidence == pb.confidence
    other = Perceptor(seed=8)
    assert any(not np.array_equal(a(TRUTH).pos, other(TRUTH).pos) for _ in range(20)), \
        "a different seed produced the same noise"

    passed = Perceptor(seed=100)(TRUTH, rng=np.random.default_rng(3))
    twin = Perceptor(seed=200)(TRUTH, rng=np.random.default_rng(3))
    assert np.array_equal(passed.pos, twin.pos) and passed.confidence == twin.confidence, \
        "a passed rng must drive the noise, not the perceptor's own"

    p = Perceptor(seed=42)
    for _ in range(5):
        p(TRUTH)
    p.reset(seed=99)
    fresh = Perceptor(seed=99)
    for _ in range(4):
        assert np.array_equal(p(TRUTH).pos, fresh(TRUTH).pos), "reset must restart the stream"
    print("[ok] noise: reproducible per seed, per passed rng, and re-seeded by reset()")


def check_occlusion_cost():
    p = Perceptor(seed=0)
    assert p.sigma(False) == p.sigma_clear and p.sigma(True) == p.sigma_occluded
    ratio = p.sigma(True) / p.sigma(False)
    assert ratio >= 8.0, f"occlusion costs only {ratio:.1f}x sigma (PLAN 2.5 wants an order)"

    draws = 400
    err, invalid, conf, dropped = {}, {}, {}, {}
    for occluded in (False, True):
        q = Perceptor(seed=42, sigma_clear=0.02, sigma_occluded=0.2, miss_rate=0.4)
        got = [q(TRUTH, occluded=occluded) for _ in range(draws)]
        err[occluded] = float(np.mean([np.linalg.norm(g.pos - OBJ) for g in got]))
        dropped[occluded] = [g for g in got if not g.valid]
        invalid[occluded] = len(dropped[occluded])
        conf[occluded] = float(np.mean([g.confidence for g in got]))
        assert (invalid[occluded] == 0) == (not occluded), "only occlusion may drop detections"
    assert invalid[True] > 0, "occlusion never dropped a detection"
    assert all(g.confidence < 0.5 for g in dropped[True]), dropped[True][:3]
    assert err[True] > 3.0 * err[False], (err[True], err[False])
    assert conf[True] < conf[False], (conf[False], conf[True])
    print(f"[ok] perceptor: mean |err| {err[False] * 1000:.1f} mm clear vs "
          f"{err[True] * 1000:.1f} mm occluded, sigma {p.sigma(False) * 1000:.0f} -> "
          f"{p.sigma(True) * 1000:.0f} mm, {invalid[True]}/{draws} occluded ticks reported no "
          f"detection, mean confidence {conf[False]:.2f} -> {conf[True]:.2f}")


def check_compressor_sections():
    comp, ring = Compressor(QLO, QHI), RingBuffer()
    mid = (QLO + QHI) / 2  # q = 0 is not mid-travel for every joint, so aim at the range centre
    prop = Proprio(t=0.0, q=mid, qvel=np.full(NJNT, 0.5), tau=np.full(NJNT, -0.2),
                   ee_pos=np.array([0.0, -0.2, 0.15]), ee_axis=np.array([0.0, -1.0, 0.0]))
    ring.push(prop)
    vec = comp(ring, prop, Percept(pos=OBJ.copy(), valid=True, confidence=0.83))
    assert vec.shape == (OBS_DIM,), (vec.shape, OBS_DIM)
    assert np.all(np.isfinite(vec))
    for name, value in (("q", prop.q), ("qvel", prop.qvel), ("tau", prop.tau),
                        ("ee_pos", prop.ee_pos), ("ee_axis", prop.ee_axis), ("obj_pos", OBJ)):
        assert np.allclose(section(vec, name), value), name
    assert section(vec, "obj_conf")[0] == 0.83
    assert np.allclose(section(vec, "joint_margin"), 1.0), "mid-travel should saturate the margin"

    at_stop = Proprio(t=TICK_DT, q=np.where(np.arange(NJNT) == NJNT - 1, QHI[-1] - 0.01, QHI))
    ring.push(at_stop)
    vec = comp(ring, at_stop, Percept(pos=GRASP.copy(), valid=False, confidence=0.05))
    margin = section(vec, "joint_margin")
    assert vec.shape == (OBS_DIM,)
    assert abs(margin[-1] - 0.02) < 1e-12, margin
    assert np.all(margin[:-1] == 0.0), margin
    assert section(vec, "obj_conf")[0] == 0.0, "a dropped detection must not read as a vague pose"
    print(f"[ok] compressor: {OBS_DIM}-float vector per tick, joint_margin at the stops "
          f"{np.array2string(margin, precision=3)}")


def check_grip_slip():
    def still(k):
        return GRASP

    def block(k):
        return OBJ  # 90 mm out along the approach axis, where a held block sits

    def carried_obj(k):
        return OBJ + [0.0, k * 0.002, 0.0]

    def carried_hand(k):
        return GRASP + [0.0, k * 0.002, 0.0]

    # Slips are radial: the distance between the block estimate and the grasp point is what the
    # index measures, and motion across that line changes it only to second order.
    def creeping(k):
        return OBJ + [0.0, k * 0.0004, 0.0]

    def tightening(k):
        return OBJ - [0.0, k * 0.0004, 0.0]

    def slip_of(vec):
        return float(section(vec, "grip_slip")[0])

    def quiet_run(obj_of, grasp_of, perceptor=None, steps=40):
        comp, ring = Compressor(QLO, QHI), RingBuffer()
        vectors = run_ticks(perceptor or clean_perceptor(seed=1), comp, ring, steps,
                            grasp_of, obj_of)
        return slip_of(vectors[-1]), comp, ring, vectors

    creep_index = (0.0004 / TICK_DT) / SLIP_LIMIT  # 20 mm/s of drift, as the vector reports it
    still_value, comp, ring, _ = quiet_run(block, still)
    carry, _, _, _ = quiet_run(carried_obj, carried_hand)
    creep, comp, ring, _ = quiet_run(creeping, still, steps=SLIP_TICKS)
    tighten, _, _, _ = quiet_run(tightening, still)
    assert abs(still_value) < 0.02, f"a block at rest in the hand reads slip {still_value}"
    assert abs(carry) < 0.02, f"carrying the block with the hand reads slip {carry}"
    assert 0.5 * creep_index < creep < 1.5 * creep_index, (creep, creep_index)
    assert -1.5 * creep_index < tighten < -0.5 * creep_index, (tighten, creep_index)

    # A 20 ms difference of a 10 mm-noise pose is ~0.5 m/s of pure noise: the window is what makes
    # slip observable at all, so a sigma the raw difference cannot survive must be filtered out.
    noisy = Perceptor(seed=5, sigma_clear=0.01, sigma_occluded=0.01, miss_rate=0.0, conf_jitter=0.0)
    loud, _, _, _ = quiet_run(block, still, perceptor=noisy)
    assert abs(loud) < 0.1, f"10 mm pose noise fakes a slip of {loud}"

    # Losing the detection freezes the estimate rather than inventing a jump: no new pixels, no new
    # information.
    for k in range(SLIP_TICKS, SLIP_TICKS + 30):
        prop = Proprio(t=k * TICK_DT, grasp_pos=np.asarray(still(k), dtype=np.float64))
        ring.push(prop)
        vec = comp(ring, prop, Percept(pos=OBJ + [0.0, 0.02, 0.0], valid=False, confidence=0.04))
    after = slip_of(vec)
    assert abs(after - creep) < 0.05, (creep, after)
    assert section(vec, "obj_conf")[0] == 0.0
    print(f"[ok] grip_slip (index, saturating at {SLIP_LIMIT * 1000:.0f} mm/s): block at rest "
          f"{still_value:+.3f}, carried rigidly with the hand {carry:+.3f}, 20 mm/s creep "
          f"{creep:+.3f}, jaw take-up {tighten:+.3f}, 10 mm-noise pose {loud:+.3f}, across a "
          f"30-tick dropout {creep:+.3f} -> {after:+.3f}")


def check_eventlog():
    path = os.path.join(tempfile.mkdtemp(prefix="spinjev_bus_"), "decisions.jsonl")
    comp, ring = Compressor(QLO, QHI), RingBuffer()
    prop = Proprio(t=0.02, q=np.zeros(NJNT))
    ring.push(prop)
    obs = comp(ring, prop, Percept(pos=OBJ.copy(), valid=True, confidence=0.9))

    model = Decision(episode_id=11, tick=1, obs=obs, candidates=("approach", "close"),
                     chosen="approach", probs={"approach": 0.62, "close": 0.38},
                     escalate=False, escalate_prob=0.11, truth=TRUTH, t=prop.t)
    expert = Decision(episode_id=11, tick=2, obs=obs, candidates=("approach", "close", "lift"),
                      chosen="close", probs={"approach": 0.0, "close": 1.0, "lift": 0.0},
                      escalate=False, truth=TRUTH, t=prop.t)
    gave_up = Decision(episode_id=11, tick=3, obs=obs, candidates=("approach",), chosen="",
                       probs={"approach": 1.0}, escalate=True, escalate_prob=0.81, outcome=1,
                       truth=TRUTH, t=prop.t)
    with EventLog(path) as log:
        for rec in (model, expert, gave_up):
            log.write(rec)
        log.flush()
    with EventLog(path) as log:
        log.write(Decision(episode_id=12, tick=0, obs=obs, candidates=("approach",),
                           chosen="approach", probs={"approach": 1.0}, outcome=0, truth=TRUTH))

    lines = list(read(path))
    assert len(lines) == 4, f"append-only writer lost or duplicated rows: {len(lines)}"
    assert all(rec["schema"] == SCHEMA_VERSION for rec in lines), lines[0]["schema"]
    assert [rec["episode_id"] for rec in lines] == [11, 11, 11, 12]
    assert [rec["outcome"] for rec in lines] == [None, None, 1, 0]
    assert [rec["escalate"] for rec in lines] == [False, False, True, False]
    assert all(isinstance(rec["escalate"], bool) for rec in lines)
    assert lines[2]["esc_prob"] == 0.81 and lines[0]["esc_prob"] == 0.11
    assert "esc_prob" not in lines[1], lines[1].keys()
    assert lines[1]["probs"] == {"approach": 0.0, "close": 1.0, "lift": 0.0}
    assert abs(sum(lines[0]["probs"].values()) - 1.0) < 1e-9, lines[0]["probs"]
    assert np.allclose(lines[0]["obs"], obs, atol=1e-5) and len(lines[0]["obs"]) == OBS_DIM
    assert np.allclose(lines[0]["truth"]["obj_pos"], OBJ, atol=1e-4)
    assert np.allclose(lines[0]["truth"]["obj_vel"], TRUTH.obj_vel, atol=1e-4)
    assert lines[0]["truth"]["ncontact"] == 3
    assert lines[0]["candidates"] == ["approach", "close"] and lines[0]["chosen"] == "approach"
    assert lines[2]["chosen"] == "" and lines[2]["outcome"] == 1

    def malformed(**kw):
        return Decision(episode_id=0, tick=0, obs=obs, **kw)

    for bad in (
        malformed(chosen="approach", probs={"approach": 0.5}),
        malformed(chosen="approach", probs={"approach": 0.6, "close": 0.3}),
        malformed(chosen="lift", probs={"approach": 1.0}),
        malformed(probs={}, escalate=True),
        malformed(chosen="approach", probs={"approach": 1.0}, outcome=2),
    ):
        try:
            bad.to_line()
        except ValueError:
            continue
        raise AssertionError(f"a broken record was accepted: {bad}")

    per_row = os.path.getsize(path) / len(lines)
    print(f"[ok] event log: v{SCHEMA_VERSION} JSONL, {len(lines)} rows round-tripped (escalate "
          f"flag + prob, one-hot and soft distributions, 0/1 outcome, privileged truth), "
          f"{per_row:.0f} B/row -> {per_row * 1e6 / 1e9:.2f} GB per million decisions")


def main():
    check_layout()
    check_ring()
    check_reproducible()
    check_occlusion_cost()
    check_compressor_sections()
    check_grip_slip()
    check_eventlog()
    print("\n[PASS] bus: perceptor, ring buffer, compressor and event log honour OBS_DIM and the "
          "observation contract.")
    print(f"       window {WINDOW_TICKS} ticks, noise sigma exposed for evaluation, nothing on "
          f"the observation path reads Truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
