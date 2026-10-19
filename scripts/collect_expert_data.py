#!/usr/bin/env python3
"""Collect decision samples from the scripted experts (PLAN 阶段 1c).

One process owns one shard and writes one JSONL file; shards are never shared between processes, so
20 array tasks can run without a lock. `(seed, episode index)` alone decides a trajectory: every
random stream is derived from that pair -- the task layer's draw for placement and friction, the
perceptor's noise, the expert's per-episode routing threshold -- so shard boundaries and episode
order cannot change a sample.

    .venv/bin/python scripts/collect_expert_data.py --selfcheck
    .venv/bin/python scripts/collect_expert_data.py --episodes 15 --seed 1 --shard 0 \
        --expert reach,direct,escalate --out data/
    .venv/bin/python scripts/collect_expert_data.py --merge --out data/ --merged data/expert.jsonl

A *sample* is one decision event, not one 50 Hz tick: the task layer only lets a new primitive be
commanded at a switch, so `PickPlace.drive()` (one primitive, run to completion) is exactly the
decision granularity. Events: `switch` (a primitive was commanded and driven), `hold` (the head
commanded nothing -- a rejected waypoint or a replan), `escalate`, plus the last record of an
episode, marked `terminal`, whose event name says how it ended (`terminal` / `guard_trip` /
`budget`). Rewards are PLAN 2.6's, read off `Outcome.reward`; `ep.reward` on every record is the
trajectory return, which is what BC conditions on.

The observation vector is the bus compressor's, over the 2 s window that
`recording.RecordingEnv` pushes every tick, so `recording.snapshot` hands the expert exactly the
vector the student will train on -- not a second renderer built for the sampling path. `Truth`
reaches a record only through its privileged `truth` field. The identity
perceptor (`--perceptor identity`, the path the task layer takes when no perception module is
wired) does feed the raw simulated pose into the object channel, so it is diagnostic only -- the
records say which perceptor made them.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from spinjev.backends.sim.mujoco_sim import TABLE_SCENE, Sim  # noqa: E402
from spinjev.bus.eventlog import SCHEMA_VERSION  # noqa: E402
from spinjev.bus.schema import OBS_DIM, OBS_LAYOUT  # noqa: E402
from spinjev.experts import EXPERTS  # noqa: E402
from spinjev.experts.perceptor_stub import StubPerceptor  # noqa: E402
from spinjev.experts.recording import (  # noqa: E402
    DATASET,
    HOLD_TICKS,
    RecordingEnv,
    decision_record,
    selfcheck,
    snapshot,
    truth_record,
)
from spinjev.tasks.pick_place import TaskSpec  # noqa: E402

#: TaskSpec's own defaults, which sit in the middle of the basin test/test_spine.py measures
#: (depth 0.085-0.095, lateral -0.035..-0.025). The margin matters: the grasp wrist goal is
#: `block_z + grasp_depth` and `Prefilter` rejects any waypoint below `min_wrist_z`, so at depth
#: 0.085 the goal lands exactly on the 0.115 fence -- a sub-millimetre sink of the block under
#: gravity flips it to "table" and every expert reports a reachable placement as unwinnable
#: (measured in the first pilot: 3/3 winnable episodes escalated). 0.090 buys 5 mm of that margin.
DEFAULT_GRASP_DEPTH = 0.090
DEFAULT_GRASP_LATERAL = -0.030
#: Carry height: left at the task layer's own default, and the sweep says that is a real choice.
#: At 0.15 the pinch drags the block to the drop zone and the placement completes, but the block
#: centre only rises ~16 mm, so `hold_z` (rest + 20 mm) is never cleared and PLAN 2.6's +1 never
#: lands. At 0.16-0.17 the block does clear `hold_z` (12-13/20 episodes get the +1) and then slips
#: out of the jaws during the carry (same 12-13/20 end "dropped", 0-1/20 placed). The two reward
#: channels are mutually exclusive on this asset today; raising the carry buys the +1 and spends
#: the placement. Nothing here tunes that away -- `--lift-height` exposes it.
DEFAULT_LIFT_HEIGHT = 0.15
#: Hard cap on decision events per episode, so a head that keeps holding still terminates.
DEFAULT_MAX_EVENTS = 30
#: Rest pose (start and retreat waypoint). The task layer's own (0, -0.26, 0.16) parks the
#: *fingertips* ~9 cm in front of the wrist, i.e. inside the band the sampler drops blocks in, so
#: the arm starts the episode already colliding with the object. Measured over 40 episodes with the
#: reach expert, everything else at task defaults:
#:     home_z 0.16 -> 17/40 placed,  5/40 fatal table trips, 16 escalates (7 of them wrong)
#:     home_z 0.20 -> 29/40 placed,  0/40 fatal table trips, 11 escalates (2 of them wrong)
#:     home_z 0.22 -> 29/40 placed,  0/40 trips (identical outcome, ~7% slower)
#: 0.20 m is the ceiling of the wrist envelope measured in test/test_spine.py, so this is not a
#: licence to reach somewhere the arm cannot hold; it is the top of the band it was already in.
DEFAULT_HOME_Z = 0.16
#: Hover clearance above the block before the descent.
DEFAULT_PREGRASP_CLEAR = 0.06


def spec_dict(spec):
    """JSON-safe view of the TaskSpec, so a shard says exactly which physics it was collected on."""
    out = {}
    for key, value in vars(spec).items():
        if key == "reward":
            out[key] = {k: float(v) for k, v in vars(value).items()}
        elif isinstance(value, np.ndarray):
            out[key] = [round(float(v), 5) for v in value]
        else:
            out[key] = value.item() if isinstance(value, np.generic) else value
    return out


def take(env, expert, shard, seed, episode_index, event_index, event, terminal, dp, decision):
    """Freeze one decision point + one decision into one record."""
    truth = env.truth()
    rec = decision_record(
        shard=shard, seed=seed, episode_index=episode_index, decision=event_index,
        expert=expert.name, event=event, terminal=terminal, dp=dp, choice=decision,
        book=expert.book(),
    )
    rec["ep"] = {
        "object_xy": [round(float(v), 5) for v in env.ep.object_xy],
        "radius": round(float(env.ep.radius), 4),
        "unreachable": bool(env.ep.unreachable),
        "occluded": bool(env.ep.occluded),
        "escalate_gt": bool(env.ep.escalate_gt),
    }
    rec["executed"] = event == "switch"
    if event_index == 0:
        # The placement itself was unworkable before the arm did anything, as opposed to made
        # unworkable by the arm (a block knocked onto its side by the transit is a real failure,
        # but it is not the sampler's unwinnable branch and must not be counted as one).
        rec["ep"]["grasp_blocked_at_first"] = bool(dp.blocked("grasp") or not dp.percept.valid)
    rec["truth"] = truth_record(truth)
    return rec


def record_episode(sim, spec, args, expert_name, seed, shard, episode_index, reset_kw=None):
    """Run one episode. Returns (records, env) -- the env is the caller's readout, not a leak.

    `reset_kw` is for probes and evaluation: pass `xy=` / `unreachable=` / `occluded=` to pin a
    placement instead of letting the sampler draw one. Collection never uses it.
    """
    ep_seed = [int(seed), int(shard) * 100000 + int(episode_index)]
    perceptor = None if args.perceptor == "identity" else StubPerceptor(
        seed=int(ep_seed[1]) + 1000003 * int(ep_seed[0]),
        sigma=args.obj_sigma, occluded_confidence=args.occluded_conf,
    )
    env = RecordingEnv(sim=sim, spec=spec, perceptor=perceptor, seed=ep_seed)
    ep = env.reset(**(reset_kw or {}))     # the sampler draws placement / branch / friction
    if perceptor is not None:
        perceptor.reset()                  # new episode: the tracker has not seen anything yet
    expert = EXPERTS[expert_name](seed=int(ep_seed[0]) * 1000003 + int(ep_seed[1]),
                                 temperature=args.temperature)
    expert.reset(ep, episode_index)

    records = []
    for event_index in range(args.max_events):
        terminal = env.out.done
        dp = snapshot(env, expert)         # the state the head sees -- before it acts
        decision = expert.decide(dp)
        action = decision.action
        if terminal:
            event = "guard_trip" if (env.out.reason or "").startswith("guard:") else "terminal"
            records.append(take(env, expert, shard, seed, episode_index, event_index,
                                event, True, dp, decision))
            break
        if action.escalate:
            records.append(take(env, expert, shard, seed, episode_index, event_index,
                                "escalate", True, dp, decision))
            env.finish(escalated=True, reason=f"{expert.name}:report")
            break
        event = "switch" if action.primitive is not None else "hold"
        records.append(take(env, expert, shard, seed, episode_index, event_index,
                            event, False, dp, decision))
        if event == "switch":
            # The drive is the *consequence* of the recorded decision, so the state the next
            # record sees is genuinely the one this action produced.
            expert.report(env.drive(action.primitive, action.goal))
        else:
            env.wait(HOLD_TICKS)
    else:
        dp = snapshot(env, expert)
        records.append(take(env, expert, shard, seed, episode_index, args.max_events,
                            "budget", True, dp, expert.decide(dp)))
        env.finish(escalated=False, reason="budget")

    out = env.out
    label = {
        "success": bool(out.success), "escalated": bool(out.escalated),
        "released": bool(out.released), "dropped": bool(out.dropped),
        "tripped": list(out.tripped), "held_ticks": int(out.held_ticks),
        "ticks": int(out.ticks), "reward": round(float(out.reward), 4),
        "reason": out.reason or "unfinished",
        "escalate_correct": bool(out.escalated) == bool(env.ep.escalate_gt),
    }
    for i, rec in enumerate(records):
        rec["ep"].update(label)
        rec["episode_reward"] = label["reward"]
        # Event-local credit: the reward table of PLAN 2.6 pays on outcomes and on the escalate
        # choice, so the only records that can carry a nonzero term are the ones that ended the
        # episode (or a hold/switch that tripped a guard mid-drive, which is the terminal record).
        rec["event_reward"] = label["reward"] if rec["terminal"] and i == len(records) - 1 else 0.0
    return records, env


# -- collect ---------------------------------------------------------------------------
def collect(args):
    experts = [name.strip() for name in args.expert.split(",") if name.strip()]
    for name in experts:
        if name not in EXPERTS:
            raise SystemExit(f"unknown expert {name!r}: have {sorted(EXPERTS)}")
    spec = TaskSpec(
        grasp_depth=args.grasp_depth, grasp_lateral=args.grasp_lateral,
        min_wrist_z=args.min_wrist_z, lift_height=args.lift_height,
        pregrasp_clear=args.pregrasp_clear, home=np.array([0.0, -0.26, args.home_z]),
        max_ticks=args.max_ticks,
    )
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    sim = Sim(xml=TABLE_SCENE)             # one model per process, reused by every episode
    loaded = time.time() - t0
    grand = 0
    for name in experts:
        rows = []
        success = 0
        t1 = time.time()
        for i in range(args.episodes):
            records, env = record_episode(sim, spec, args, name, args.seed, args.shard, i)
            rows.extend(records)
            success += sum(1 for r in records if r["terminal"] and r["ep"]["success"])
        path = os.path.join(args.out, f"expert_{name}_shard{args.shard:03d}.jsonl")
        write_shard(path, rows, args, name, spec)
        grand += len(rows)
        wall = time.time() - t1
        eps = stats_of(rows)
        print(f"[shard {args.shard}] {name:<9} {args.episodes} ep -> {len(rows)} samples "
              f"({len(rows) / max(args.episodes, 1):.2f}/ep) {success}/{args.episodes} terminal-success "
              f"({success / max(args.episodes, 1) * 100:.1f}%) "
              f"unwinnable {eps['unwinnable']}/{args.episodes} escalate {eps['escalated']} "
              f"| {wall:.1f} s, {wall / max(args.episodes, 1) * 1000:.0f} ms/ep, "
              f"{len(rows) / max(wall, 1e-9):.1f} samples/s -> {path}")
    print(f"[shard {args.shard}] done {grand} samples, {len(experts) * args.episodes} episodes, "
          f"{time.time() - t0:.1f} s wall ({loaded:.1f} s model load)")
    return grand


def stats_of(rows):
    """Cheap per-shard counters for the progress line."""
    eps = {r["ep"]["reason"]: 0 for r in rows if r["terminal"]}
    return {
        "unwinnable": sum(1 for r in rows if r["terminal"] and (r["ep"]["unreachable"] or r["ep"]["occluded"])),
        "escalated": sum(1 for r in rows if r["terminal"] and r["ep"]["escalated"]),
        "reasons": eps,
    }


def write_shard(path, rows, args, expert_name, spec):
    meta = {
        "dataset": DATASET, "v": SCHEMA_VERSION, "kind": "shard", "expert": expert_name,
        "shard": int(args.shard),
        "seed": int(args.seed), "episodes": int(args.episodes), "samples": len(rows),
        "mujoco": mujoco.__version__, "python": sys.version.split()[0], "numpy": np.__version__,
        "obs_layout": [list(x) for x in OBS_LAYOUT], "obs_dim": int(OBS_DIM),
        "task": spec_dict(spec), "perceptor": args.perceptor, "obj_sigma": float(args.obj_sigma),
        "occluded_conf": float(args.occluded_conf), "temperature": float(args.temperature),
        "max_events": int(args.max_events), "scene": os.path.basename(TABLE_SCENE),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(meta) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


# -- merge + summary -------------------------------------------------------------------
def read_shards(paths):
    rows, metas = [], []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (metas if rec.get("kind") in ("shard", "merged") else rows).append(rec)
    return rows, metas


def summarise(rows):
    """Coverage and correctness tables: per-expert success, the escalate branch split four ways,
    and the candidate-set size distribution PLAN 2.3 says the pre-filter has to actually constrain."""
    out = {"samples": len(rows), "events": {}, "candidates": {}, "experts": {}}
    episodes = {}
    per_episode = {}
    for rec in rows:
        out["events"][rec["event"]] = out["events"].get(rec["event"], 0) + 1
        size = int(rec["candidate_size"])
        out["candidates"][size] = out["candidates"].get(size, 0) + 1
        key = (rec["shard"], rec["expert"], rec["episode_index"])
        per_episode[key] = per_episode.get(key, 0) + 1
        if rec["terminal"]:
            episodes[key] = rec if rec["terminal"] else episodes.get(key)
    out["episodes"] = len(episodes)
    for (shard, name, index), rec in episodes.items():
        e = out["experts"].setdefault(name, {
            "episodes": 0, "samples": 0, "success": 0, "escalated": 0, "escalate_correct": 0,
            "wrong_escalate": 0, "missed_escalate": 0, "unwinnable": 0, "guard_trips": 0,
            "dropped": 0, "timeouts": 0, "reward_sum": 0.0, "reasons": {}, "branches": {},
            "winnable": 0, "success_on_winnable": 0, "hold_reward": 0,
        })
        ep = rec["ep"]
        e["episodes"] += 1
        e["samples"] += per_episode[(shard, name, index)]
        unwinnable = bool(ep["unreachable"] or ep["occluded"])
        e["unwinnable"] += int(unwinnable)
        branch = "away" if ep["unreachable"] else ("occluded" if ep["occluded"] else "reachable")
        e["branches"][branch] = e["branches"].get(branch, 0) + 1
        e["reasons"][ep["reason"]] = e["reasons"].get(ep["reason"], 0) + 1
        e["winnable"] += int(not unwinnable)
        e["success_on_winnable"] += int(ep["success"] and not unwinnable)
        e["hold_reward"] += int(ep["held_ticks"] >= 25)
        e["success"] += int(ep["success"])
        e["escalated"] += int(ep["escalated"])
        e["escalate_correct"] += int(ep["escalated"] and ep["escalate_gt"])
        e["wrong_escalate"] += int(ep["escalated"] and not ep["escalate_gt"])
        e["missed_escalate"] += int(ep["escalate_gt"] and not ep["escalated"])
        e["guard_trips"] += int(ep["reason"].startswith("guard:"))
        e["dropped"] += int(ep["dropped"])
        e["timeouts"] += int(ep["reason"] in ("timeout", "budget", "unfinished"))
        e["reward_sum"] += float(ep["reward"])
    for name, e in out["experts"].items():
        n = max(e["episodes"], 1)
        e["samples_per_episode"] = round(e["samples"] / n, 2)
        e["success_rate"] = round(e["success"] / n, 4)
        e["success_rate_on_winnable"] = round(e["success_on_winnable"] / max(e["winnable"], 1), 4)
        e["hold_rate"] = round(e["hold_reward"] / n, 4)
        e["escalate_rate"] = round(e["escalated"] / n, 4)
        e["escalate_correct_rate"] = round(e["escalate_correct"] / max(e["escalated"], 1), 4)
        e["unwinnable_share"] = round(e["unwinnable"] / n, 4)
        e["mean_return"] = round(e["reward_sum"] / n, 4)
        e["wrong_escalate_rate"] = round(e["wrong_escalate"] / n, 4)
    return out


def print_summary(stats, wall=None, indent=""):
    print(f"{indent}samples {stats['samples']}   episodes {stats['episodes']}"
          + (f"   {stats['samples'] / wall:.1f} samples/s" if wall else ""))
    print(f"{indent}events {stats['events']}")
    print(f"{indent}candidate-set sizes {dict(sorted(stats['candidates'].items()))}")
    for name, e in sorted(stats["experts"].items()):
        print(f"{indent}{name:<9} ep {e['episodes']:>5}  samples {e['samples']:>6} "
              f"({e['samples_per_episode']}/ep)  success {e['success_rate'] * 100:5.1f}% "
              f"(on winnable {e['success_rate_on_winnable'] * 100:5.1f}%, hold+1 {e['hold_rate'] * 100:4.1f}%)  "
              f"escalate {e['escalate_rate'] * 100:5.1f}% of which correct "
              f"{e['escalate_correct_rate'] * 100:5.1f}%  wrong {e['wrong_escalate']}  "
              f"missed {e['missed_escalate']}  unwinnable {e['unwinnable_share'] * 100:4.1f}%  "
              f"guard trips {e['guard_trips']}  mean return {e['mean_return']:+.3f}")
        print(f"{indent}{'':<9}branches {e['branches']}  reasons {e['reasons']}")


def merge(args):
    paths = sorted(glob.glob(os.path.join(args.out, args.pattern)))
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(args.merged or "")]
    if not paths:
        raise SystemExit(f"no shards matched {os.path.join(args.out, args.pattern)}")
    rows, metas = read_shards(paths)
    if args.merged:
        tmp = args.merged + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(json.dumps({"dataset": DATASET, "v": SCHEMA_VERSION, "kind": "merged",
                                 "samples": len(rows),
                                 "shards": len(paths)}) + "\n")
            for rec in rows:
                fh.write(json.dumps(rec) + "\n")
        os.replace(tmp, args.merged)
        print(f"[merge] wrote {args.merged} ({len(rows)} samples from {len(paths)} shards)")
    stats = summarise(rows)
    stats["shards"] = len(paths)
    stats["mujoco"] = sorted({str(m.get("mujoco", "?")) for m in metas})
    stats["perceptors"] = sorted({str(m.get("perceptor", "?")) for m in metas})
    stats["temperatures"] = sorted({float(m.get("temperature", 0.0)) for m in metas})
    print(json.dumps({k: v for k, v in stats.items() if k != "experts"}, indent=2, default=str))
    print_summary(stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per expert in this shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="data", help="shard directory (one file per process)")
    ap.add_argument("--expert", default="reach,direct,escalate", help="comma-separated expert names")
    ap.add_argument("--max-ticks", type=int, default=900,
                    help="decision ticks per episode (TaskSpec.max_ticks, 900 = 18 s at 50 Hz)")
    ap.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS,
                    help="cap on decision *events* per episode")
    ap.add_argument("--temperature", type=float, default=0.80, help="Choice-head softmax temperature")
    ap.add_argument("--grasp-depth", type=float, default=DEFAULT_GRASP_DEPTH)
    ap.add_argument("--grasp-lateral", type=float, default=DEFAULT_GRASP_LATERAL)
    ap.add_argument("--min-wrist-z", type=float, default=0.115)
    ap.add_argument("--lift-height", type=float, default=DEFAULT_LIFT_HEIGHT)
    ap.add_argument("--pregrasp-clear", type=float, default=DEFAULT_PREGRASP_CLEAR,
                    help="hover height above the block")
    ap.add_argument("--home-z", type=float, default=DEFAULT_HOME_Z,
                    help="rest-pose wrist height (the start and retreat waypoint)")
    ap.add_argument("--perceptor", default="stub", choices=("stub", "identity"),
                    help="stub = noise + a detection flag for the occluded branch; identity = the "
                         "task layer's raw pose (diagnostic only: it puts privileged state in the "
                         "object channel and makes the occluded branch undecidable)")
    ap.add_argument("--obj-sigma", type=float, default=0.0,
                    help="perceptor noise, metres per axis. The dataset ships at 0.0 because the "
                         "scripted experts do not survive more: 0.005 halves the pick rate and "
                         "0.020 empties it (see spinjev/experts/perceptor_stub.py).")
    ap.add_argument("--occluded-conf", type=float, default=0.25,
                    help="confidence reported when the object is behind the occluder")
    ap.add_argument("--merge", action="store_true", help="merge shards in --out and print the summary")
    ap.add_argument("--merged", default=None, help="write the merged JSONL here (--merge only)")
    ap.add_argument("--pattern", default="expert_*.jsonl", help="glob for --merge")
    ap.add_argument("--selfcheck", action="store_true", help="assert the iron rules and exit")
    ap.add_argument("--summary-json", default=None, help="also dump the merge summary here")
    args = ap.parse_args()

    if args.selfcheck:
        print(json.dumps(selfcheck(), indent=2))
        return 0
    if args.merge:
        stats = merge(args)
        if args.summary_json:
            with open(args.summary_json, "w") as fh:
                json.dump(stats, fh, indent=2, default=str)
            print(f"[merge] summary -> {args.summary_json}")
        return 0
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    collect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
