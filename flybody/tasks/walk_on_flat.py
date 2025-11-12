"""Task of tethered fly walking on flat ground (no floating ball)."""
# ruff: noqa: F821

from typing import Optional
import numpy as np

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.utils import rewards

from flybody.tasks.base import Walking
from flybody.tasks.constants import (_TERMINAL_ANGVEL, _TERMINAL_LINVEL)


class WalkOnFlat(Walking):
    """Untethered fly walking on flat ground (59-DOF control)."""

    def __init__(self, claw_friction: Optional[float] = 2.0, **kwargs):
        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)

        # ------------------------------------------------------------
        # 🔓 Untether the fly - keep freejoint on attachment_frame (top level)
        # ------------------------------------------------------------
        # The attachment_frame already has a freejoint at top level (created by composer.attach)
        # We keep it (unlike walk_on_ball which removes it to tether the fly)
        # DO NOT add another one - that would cause "more than 6 dofs" error
        attachment = self.root_entity.mjcf_model.find('attachment_frame', 'walker')
        if attachment is not None:
            # Verify freejoint exists (should be created automatically by composer)
            if hasattr(attachment, 'freejoint') and attachment.freejoint is not None:
                print("[WalkOnFlat] Free joint exists on attachment_frame (untethered).")
            else:
                print("[WalkOnFlat] WARNING: No free joint found on attachment_frame!")
        else:
            print("[WalkOnFlat] WARNING: Could not find attachment_frame.")

        # ------------------------------------------------------------
        # 🦾 Switch actuator mode to force control (muscle-like)
        # ------------------------------------------------------------
        if not getattr(self._walker, "_force_mode_applied", False):
            for actuator in self._walker.mjcf_model.find_all('actuator'):
                if actuator.tag not in ['adhesion']:
                    try:
                        actuator.set_attributes(
                            biastype=None,
                            dyntype="none",
                            gainprm=[1, 0, 0, 0, 0],
                            biasprm=[0, 0, 0],
                            ctrlrange=[-1, 1]
                        )
                        if hasattr(actuator, "gear"):
                            actuator.gear = [1]
                    except Exception as e:
                        print(f"[WalkOnFlat] Skipped actuator {actuator.name}: {e}")
            self._walker._force_mode_applied = True
            # print("[WalkOnFlat] Converted actuators to force (motor) mode.")

        # ------------------------------------------------------------
        # 🎯 Keep only the 59 biologically relevant actuators
        # ------------------------------------------------------------
        target_names = [
            # --- Head (3) ---
            "head_abduct", "head_twist", "head",
            # --- Abdomen (2) ---
            "abdomen_abduct", "abdomen",
            # --- Legs (6x8 = 48) ---
            "coxa_abduct_T1_left", "coxa_twist_T1_left", "coxa_T1_left", "femur_twist_T1_left", "femur_T1_left", "tibia_T1_left", "tarsus_T1_left", "tarsus2_T1_left",
            "coxa_abduct_T1_right", "coxa_twist_T1_right", "coxa_T1_right", "femur_twist_T1_right", "femur_T1_right", "tibia_T1_right", "tarsus_T1_right", "tarsus2_T1_right",
            "coxa_abduct_T2_left", "coxa_twist_T2_left", "coxa_T2_left", "femur_twist_T2_left", "femur_T2_left", "tibia_T2_left", "tarsus_T2_left", "tarsus2_T2_left",
            "coxa_abduct_T2_right", "coxa_twist_T2_right", "coxa_T2_right", "femur_twist_T2_right", "femur_T2_right", "tibia_T2_right", "tarsus_T2_right", "tarsus2_T2_right",
            "coxa_abduct_T3_left", "coxa_twist_T3_left", "coxa_T3_left", "femur_twist_T3_left", "femur_T3_left", "tibia_T3_left", "tarsus_T3_left", "tarsus2_T3_left",
            "coxa_abduct_T3_right", "coxa_twist_T3_right", "coxa_T3_right", "femur_twist_T3_right", "femur_T3_right", "tibia_T3_right", "tarsus_T3_right", "tarsus2_T3_right",
            # --- Adhesion (6) ---
            "adhere_claw_T1_left", "adhere_claw_T1_right",
            "adhere_claw_T2_left", "adhere_claw_T2_right",
            "adhere_claw_T3_left", "adhere_claw_T3_right",
        ]

        kept_actuators = []
        for actuator in list(self._walker.mjcf_model.find_all('actuator')):
            if not any(name in actuator.name for name in target_names):
                actuator.remove()
            else:
                kept_actuators.append(actuator.name)

        # print(f"[WalkOnFlat] Keeping {len(kept_actuators)} actuators (target 59):")
        # for name in kept_actuators:
        #     print("  -", name)

        # Freeze joints not in target set by disabling limits or setting valid range
        for joint in list(self._walker.mjcf_model.find_all('joint')):
            if not any(name in joint.name for name in target_names):
                # Don't set range=[0,0] as it's invalid. Instead, disable limits or use tiny valid range
                if hasattr(joint, 'limited'):
                    joint.limited = False  # Remove limits to allow full range, or set a tiny valid range
                # Alternative: use tiny valid range if limited must be True
                # joint.limited = True
                # joint.range = [-1e-6, 1e-6]  # Tiny valid range

        # ------------------------------------------------------------
        # Note: Keep freejoint on attachment_frame for untethered movement
        # (walk_on_ball removes it to tether the fly, but we want it free)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Exclude self-collisions between thorax and limbs
        # ------------------------------------------------------------
        thorax_body = self._walker.mjcf_model.find('body', 'thorax')
        if thorax_body:
            for child in thorax_body.all_children():
                if child.tag == 'body':
                    self._walker.mjcf_model.contact.add(
                        'exclude',
                        name=f'thorax_{child.name}',
                        body1='thorax',
                        body2=child.name
                    )

        # ------------------------------------------------------------
        # Adjust claw friction for surface grip
        # ------------------------------------------------------------
        if claw_friction is not None:
            default_geom = self._walker.mjcf_model.find('default', 'adhesion-collision')
            if default_geom and hasattr(default_geom, 'geom'):
                default_geom.geom.friction = (claw_friction,)
            print(f"[WalkOnFlat] Set claw friction to {claw_friction}.")

        # ------------------------------------------------------------
        # Add forward velocity observable
        # ------------------------------------------------------------
        self._walker.observables.add_observable('forward_velocity', self.forward_velocity)
        print("[WalkOnFlat] Added forward_velocity observable.")



    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    @composer.observable
    def forward_velocity(self):
        """Observable: horizontal (forward) component of body velocity."""
        def get_forward_velocity(physics):
            try:
                v = physics.named.data.sensordata['walker/velocimeter']  # or 'velocimeter'
                return v[1] if len(v) > 1 else 0.0
            except KeyError:
                # fallback to zero velocity if sensor not defined
                return 0.0
        return observable.Generic(get_forward_velocity)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def get_reward_factors(self, physics):
        """Returns reward components for walking on flat ground."""
        # Forward progress
        forward_vel = self.forward_velocity(physics)
        velocity_reward = rewards.tolerance(
            forward_vel,
            bounds=(0.02, 0.3),
            margin=1.0,
            sigmoid='linear',
            value_at_margin=0.0,
        )

        # Uprightness from thorax world z-axis
        zaxis = physics.named.data.xmat['walker/thorax'][6:]  # world z-axis
        uprightness = zaxis[-1] if len(zaxis) == 9 else 1.0
        upright_reward = rewards.tolerance(
            uprightness, bounds=(0.9, 1.0), margin=0.3, sigmoid='linear'
        )

        return np.hstack([velocity_reward, upright_reward])

    # ------------------------------------------------------------------
    # Episode flow
    # ------------------------------------------------------------------
    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

    def initialize_episode(self, physics: 'mjcf.Physics', random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        # small random rotation or pose perturbation if desired
        # physics.named.data.qpos['thorax'][2] = 0.015  # start slightly above plane

    def before_step(self, physics: 'mjcf.Physics', action, random_state: np.random.RandomState):
        super().before_step(physics, action, random_state)

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Terminate if walker falls or spins out."""
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))
        return (
            linvel > _TERMINAL_LINVEL
            or angvel > _TERMINAL_ANGVEL
            or super().check_termination(physics)
        )
