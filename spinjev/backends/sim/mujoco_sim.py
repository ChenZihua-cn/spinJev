"""SO-ARM100 in MuJoCo: model loading, the 500 Hz servo interface, and state readout."""

import os

import numpy as np
import mujoco

from spinjev.backends.sim.kinematics import ArmKinematics
from spinjev.bus.schema import NJNT, Proprio, Truth

ASSET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "so_arm100"))
TABLE_SCENE = os.path.join(ASSET_DIR, "table_scene.xml")

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper_joint",
)
# All five joints that move the wrist are solved for, wrist_roll included: the orientation task
# in kinematics.py is what keeps the jaws straddling the block instead of sweeping through it.
REACH_JOINT_NAMES = ARM_JOINT_NAMES[:5]
OBJECT_JOINT = "target"
EE_BODY = "wrist_2_link"
TABLE_BODY = "table"
# Anything that moves counts as a collision candidate; `base` is bolted to the table, so its
# permanent contact patch is not an accident.
MOVING_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
                 "wrist_2_link", "jaw_link")
# Measured against table_scene.xml (test/test_spine.py --profile and the descent sweep re-derive
# them): the jaws point along wrist_2's local -y and close along its local x. The gripper is *not*
# symmetric about that axis -- the channel the block has to sit in is offset from the approach
# axis -- so driving the body origin straight onto a block tips the block over instead of catching
# it. GRASP_DEPTH/GRASP_LATERAL are the offset of the point a block must occupy to be gripped,
# measured as the centre of the basin where the descent sweep actually lifts: depth 0.085..0.095,
# lateral -0.035..-0.025.
EE_AXIS_LOCAL = (0.0, -1.0, 0.0)
GRASP_DEPTH = 0.090
GRASP_LATERAL = -0.030

# Arms-up rest pose: what the joints hold between episodes. Gripper at 1.8 rad is open
# (the range runs -0.22 = shut to 2.0 = wide).
HOME_Q = np.array([0.0, 0.6, -0.8, 0.0, 0.0, 1.8])


class Sim:
    """Physics at `model.opt.timestep` (2 ms -> 500 Hz), which is also the servo rate."""

    def __init__(self, xml=TABLE_SCENE, object_xy=(0.0, -0.24)):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_joints = np.array([jid(n) for n in ARM_JOINT_NAMES])
        if self.arm_joints.min() < 0:
            raise ValueError(f"model is missing arm joints: {ARM_JOINT_NAMES}")
        self.arm_qadr = m.jnt_qposadr[self.arm_joints]
        self.arm_dofadr = m.jnt_dofadr[self.arm_joints]
        self.ujnt = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINT_NAMES]
        )
        if len(self.ujnt) != NJNT or self.ujnt.min() < 0:
            raise ValueError("model does not expose one position actuator per arm joint")
        self.gadr = int(self.arm_qadr[-1])
        self.gdof = int(self.arm_dofadr[-1])
        self.obj_addr = int(m.jnt_qposadr[jid(OBJECT_JOINT)])
        self.obj_dof = int(m.jnt_dofadr[jid(OBJECT_JOINT)])
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_JOINT)
        self.block_half_z = float(m.geom_size[gid][2]) if gid >= 0 else 0.0
        self.ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.moving_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in MOVING_BODIES}
        self.table_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY)
        self.nmocap = int(m.nmocap)
        self.qlo = m.jnt_range[self.arm_joints, 0].copy()
        self.qhi = m.jnt_range[self.arm_joints, 1].copy()
        self.tau_limit = np.abs(m.actuator_forcerange[self.ujnt, 1]).copy()
        self.kin = ArmKinematics(
            m, d, EE_BODY, REACH_JOINT_NAMES, EE_AXIS_LOCAL,
            orient_weight=1.5, posture_weight=0.03,
        )
        self.kin.qref = np.array([0.0, 0.9, -1.0, 0.05, 0.0])
        self.dt = float(m.opt.timestep)
        self.reset(object_xy)

    # -- episode setup -------------------------------------------------------------
    def reset(self, object_xy=(0.0, -0.24), home_q=HOME_Q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = home_q
        self.place_object(object_xy)
        self.park_occluder()
        self.data.ctrl[self.ujnt] = home_q
        mujoco.mj_forward(self.model, self.data)
        return self

    def place_object(self, xy, z=None):
        a = self.obj_addr
        self.data.qpos[a : a + 3] = [xy[0], xy[1], self.block_half_z if z is None else z]
        self.data.qpos[a + 3] = 1.0
        self.data.qpos[a + 4 : a + 6] = 0.0
        self.data.qvel[self.obj_dof : self.obj_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_occluder(self, pos):
        if not self.nmocap:
            raise ValueError("scene has no mocap body to move")
        self.data.mocap_pos[0] = pos
        mujoco.mj_forward(self.model, self.data)

    def park_occluder(self):
        if self.nmocap:
            self.data.mocap_pos[0] = [0.6, 0.6, 0.6]
            mujoco.mj_forward(self.model, self.data)

    # -- control -------------------------------------------------------------------
    def act(self, u):
        """Joint-angle setpoints for the position actuators, in radians."""
        self.data.ctrl[self.ujnt] = np.clip(u, self.qlo, self.qhi)

    def hold(self):
        self.act(self.data.qpos[self.arm_qadr])

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return self

    # -- readout -------------------------------------------------------------------
    def sense(self) -> Proprio:
        d = self.data
        pos, axis, close = self.kin.ee()
        return Proprio(
            t=float(d.time),
            q=d.qpos[self.arm_qadr].copy(),
            qvel=d.qvel[self.arm_dofadr].copy(),
            tau=d.actuator_force[self.ujnt].copy(),
            ee_pos=pos,
            ee_axis=axis,
            ee_close=close,
            grasp_pos=pos + axis * GRASP_DEPTH + close * GRASP_LATERAL,
        )

    def truth(self) -> Truth:
        a, v = self.obj_addr, self.obj_dof
        return Truth(
            obj_pos=self.data.qpos[a : a + 3].copy(),
            obj_quat=self.data.qpos[a + 3 : a + 7].copy(),
            obj_vel=self.data.qvel[v : v + 3].copy(),
            ncontact=int(self.data.ncon),
        )

    def static_torque(self, q):
        """Gravity torque at joint config `q`, held still: the current the servos would have to
        carry to hold that pose. Reaching a waypoint is not the same as being able to hold it."""
        m, d = self.model, self.data
        saved_q, saved_v = d.qpos[self.arm_qadr].copy(), d.qvel[self.arm_dofadr].copy()
        d.qpos[self.arm_qadr] = q
        d.qvel[self.arm_dofadr] = 0.0
        mujoco.mj_forward(m, d)
        torque = d.qfrc_bias[self.arm_dofadr].copy()
        d.qpos[self.arm_qadr] = saved_q
        d.qvel[self.arm_dofadr] = saved_v
        mujoco.mj_forward(m, d)
        return torque

    def contact_force_with_table(self) -> float:
        """Largest normal force between a moving link and the tabletop, in newtons.

        Grazing contact is *not* an accident here: picking a 3 cm block off a flat table means the
        fingertips sweep at table level. The guard reacts to pressing, which on the real arm is
        the same signal -- an unexplained current spike with nothing commanded to push against.
        """
        d, m = self.data, self.model
        worst = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            if (b1 in self.moving_bodies and b2 == self.table_body) or (
                b2 in self.moving_bodies and b1 == self.table_body
            ):
                mujoco.mj_contactForce(m, d, i, f6)
                worst = max(worst, float(np.linalg.norm(f6[:3])))
        return worst

    def touching_table(self) -> bool:
        return self.contact_force_with_table() > 0.0
