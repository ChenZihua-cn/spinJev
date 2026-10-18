"""Kinematics service for the sim backend.

`backends/robot/` must offer the same surface (encoder -> FK, goal -> joint targets) for the
spine to stay backend-agnostic.

The task is wrist position + "jaws pointing down", over the 5 arm joints, damped least squares.
Gripping needs one more thing that position alone cannot give: the jaws close along the wrist's
local x, so if the roll joint wanders the fingers sweep *through* the block instead of either
side of it (measured: the block got shoved 50 mm aside). Roll is the only joint that spins the
jaws about the approach axis, so it gets a strong posture pull of its own instead of a sixth
task row -- six rows will not fit on five DOF without costing position accuracy.
"""

import numpy as np
import mujoco

CLOSE_LOCAL = np.array([1.0, 0.0, 0.0])
DOWN = np.array([0.0, 0.0, -1.0])


class ArmKinematics:
    def __init__(self, model, data, body_name, joint_names, axis_local, orient_weight=1.5,
                 posture_weight=0.03, roll_weight=0.25, roll_index=4, roll_target=0.0,
                 step_limit=0.15, damping=0.06):
        self.m, self.d = model, data
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if self.body < 0 or any(j < 0 for j in joints):
            raise ValueError("named body/joints are not all in the model")
        self.addr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dof = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo = np.array([model.jnt_range[j][0] for j in joints])
        self.hi = np.array([model.jnt_range[j][1] for j in joints])
        self.axis_local = np.asarray(axis_local, dtype=np.float64)
        self.orient_weight = orient_weight
        self.step_limit = step_limit
        self.damping = damping
        self.qref = np.zeros(self.lo.size)
        self.posture = np.full(self.lo.size, posture_weight)
        self.roll_index = roll_index
        self.roll_target = roll_target
        if roll_index < self.lo.size:
            self.posture[roll_index] = roll_weight

    def ee(self):
        """(wrist origin, unit approach axis, closing direction) in the world frame."""
        rot = self.d.xmat[self.body].reshape(3, 3)
        return self.d.xpos[self.body].copy(), rot @ self.axis_local, rot @ CLOSE_LOCAL

    def point(self, q):
        """Wrist pose for joint config `q`, leaving the live state untouched."""
        saved = self.d.qpos[self.addr].copy()
        self.d.qpos[self.addr] = q
        mujoco.mj_forward(self.m, self.d)
        out = self.ee()
        self.d.qpos[self.addr] = saved
        mujoco.mj_forward(self.m, self.d)
        return out

    def solve(self, goal, seed, iters=60, tol=2e-3):
        """Return (q, residual, axis). Iterating from `seed` keeps one warm start per waypoint."""
        q = np.clip(seed, self.lo, self.hi)
        saved = self.d.qpos[self.addr].copy()
        w = self.orient_weight
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        err_pos = np.zeros(3)
        for _ in range(iters):
            self.d.qpos[self.addr] = q
            mujoco.mj_forward(self.m, self.d)
            pos, axis, _ = self.ee()
            err_pos = goal - pos
            if np.linalg.norm(err_pos) < tol:
                break
            mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.body)
            e = np.concatenate([err_pos, w * np.cross(axis, DOWN)])
            J = np.vstack([jacp[:, self.dof], w * jacr[:, self.dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
            dq /= max(1.0, float(np.max(np.abs(dq))) / self.step_limit)
            q = np.clip(q + dq + self.posture * (self.qref - q), self.lo, self.hi)
        self.d.qpos[self.addr] = saved
        mujoco.mj_forward(self.m, self.d)
        pos, axis, close = self.point(q)
        return q, float(np.linalg.norm(goal - pos)), axis, close

    def closing_heading(self, close):
        """Heading of the jaw closing direction, in degrees from world +x. A block sitting
        axis-aligned on the table grips at 0 +- 45 deg; anything past that is a finger in its face."""
        return float(np.degrees(np.arctan2(close[1], close[0])))
