"""MuJoCo smoke test: import, physics stepping, offscreen rendering.

Run:      python test/test_mujoco.py
Probe a specific render backend (used internally, one backend per process):
          python test/test_mujoco.py --render-only egl
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

SCENE_XML = """
<mujoco model="smoke">
  <option>
    <flag energy="enable"/>
  </option>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 4" dir="0 0 -1"/>
    <geom type="plane" size="0 0 1" rgba=".9 .9 .9 1"/>
    <body name="box" pos="0.2 0.1 1.5">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" mass="1" rgba=".2 .5 .9 1"/>
    </body>
    <camera name="cam" pos="2.5 -2.5 1.8" xyaxes="0.707 0.707 0 -0.4 0.4 0.8"/>
  </worldbody>
</mujoco>
"""

BACKENDS = ("glfw", "egl", "osmesa")


def check_import():
    import mujoco

    print(f"[ok] mujoco {mujoco.__version__} from {os.path.dirname(mujoco.__file__)}")
    print(f"     python {sys.version.split()[0]} | MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
    return mujoco


def build_model(mujoco):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return model, mujoco.MjData(model)


def check_physics(mujoco, model, data, steps):
    qpos0 = data.qpos.copy()
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - t0

    if not np.all(np.isfinite(data.qpos)):
        raise AssertionError("qpos contains NaN/Inf after stepping")
    moved = float(np.max(np.abs(data.qpos - qpos0)))
    if moved < 1e-3:
        raise AssertionError(f"qpos did not change (max delta {moved:.2e}); simulation looks inert")

    print(
        f"[ok] physics: {steps} steps in {elapsed:.3f}s ({steps / elapsed / 1000:.1f}k steps/s), "
        f"max qpos delta {moved:.3f}, z={data.qpos[2]:.3f}, "
        f"Epot={data.energy[0]:.3f}, Ekin={data.energy[1]:.3f}"
    )


def save_ppm(img, path):
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def render_once(mujoco, model, data, outdir, tag):
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam")
        img = renderer.render()
    finally:
        # Renderer.close() only exists from MuJoCo 3.1 onwards.
        close = getattr(renderer, "close", None)
        if close:
            close()
    if img.shape != (240, 320, 3):
        raise AssertionError(f"unexpected image shape {img.shape}")
    if float(img.std()) < 1.0:
        raise AssertionError(f"rendered image is blank (std={img.std():.2f})")
    out = os.path.join(outdir, f"render_{tag}.ppm")
    save_ppm(img, out)
    print(f"       mean={img.mean():.1f} std={img.std():.1f} -> {out}")


def check_backend(mujoco, model, data, outdir, backend):
    env = dict(os.environ)
    if backend is not None:
        env["MUJOCO_GL"] = backend
    else:
        env.pop("MUJOCO_GL", None)
    cmd = [sys.executable, os.path.abspath(__file__), "--render-only", backend or "auto", "--outdir", outdir]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    label = backend or "auto"
    if proc.returncode == 0:
        print(f"[ok] render backend {label}")
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip().startswith("mean="):
                print(f"       {line.strip()}")
        return label
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    print(f"[fail] render backend {label}: {tail[-1] if tail else 'no output'}")
    return None


def check_rendering(mujoco, model, data, outdir):
    probes = [os.environ.get("MUJOCO_GL")] if os.environ.get("MUJOCO_GL") else [None]
    probes += [b for b in BACKENDS if b not in probes]
    return [label for label in (check_backend(mujoco, model, data, outdir, b) for b in probes) if label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--render-only", help="internal: render one frame with the ambient MUJOCO_GL and exit")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mujoco = check_import()
    model, data = build_model(mujoco)
    check_physics(mujoco, model, data, args.steps)

    if args.render_only:
        render_once(mujoco, model, data, args.outdir, args.render_only)
        return 0
    if args.skip_render:
        print("[skip] rendering (--skip-render)")
        return 0

    working = check_rendering(mujoco, model, data, args.outdir)
    if not working:
        print("\n[FAIL] physics works but no offscreen render backend is usable.")
        print("       Headless GPU nodes usually need MUJOCO_GL=egl; CPU nodes need osmesa (libosmesa6).")
        return 1
    print(f"\n[PASS] MuJoCo is functional. Working render backends: {', '.join(working)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
