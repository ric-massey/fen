"""
Stage 1 body: a 4-DOF arm in MuJoCo, with unpredictable external perturbations.

The perturbation label is recorded for EVALUATION ONLY. It is never given to the
forward model, and nothing observable is correlated with it — that is the whole
point. If the model could predict perturbations, the self/world boundary would
collapse (see FEN.md, Stage 1 failure modes).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

WORLD = Path(__file__).parent / "world.xml"


class Body:
    def __init__(
        self,
        seed: int = 0,
        frame_skip: int = 5,
        perturb_rate: float = 0.02,      # probability per control step of a burst starting
        perturb_steps: int = 3,          # control steps a step-onset burst lasts
        perturb_force: float = 12.0,     # newtons, scale of the random force
        ramp_prob: float = 0.5,          # fraction of bursts with smooth onset
        ramp_steps: int = 12,            # burst length when ramping
    ):
        self.model = mujoco.MjModel.from_xml_path(str(WORLD))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.frame_skip = frame_skip
        self.perturb_rate = perturb_rate
        self.ramp_prob = ramp_prob
        self.step_steps = perturb_steps
        self.ramp_steps = ramp_steps
        self.perturb_force = perturb_force

        self.nq = self.model.nq          # 4
        self.nu = self.model.nu          # 4
        self.obs_dim = self.nq * 2       # positions + velocities

        # Link bodies eligible for external force.
        self.link_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("l1", "l2", "l3", "l4")
        ]

        self._burst_left = 0
        self._burst_len = 0
        self._burst_ramp = False
        self._burst_force = np.zeros(6)
        self._burst_body = self.link_ids[0]
        self._onset = False

        self.reset()

    # ── observation ──────────────────────────────────────────────────────────

    def obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[: self.nq], self.data.qvel[: self.nq]]).copy()

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.nq] = self.rng.uniform(-0.4, 0.4, self.nq)
        self.data.qvel[: self.nq] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._burst_left = 0
        return self.obs()

    # ── perturbation ─────────────────────────────────────────────────────────

    def _maybe_start_burst(self) -> None:
        """Poisson-timed, random body, random direction, random magnitude.

        Nothing about the timing, target, or direction is inferable from the
        observation or the action. That unpredictability is load-bearing.
        """
        self._onset = False
        if self._burst_left > 0:
            return
        if self.rng.random() < self.perturb_rate:
            # Mixed onsets: some bursts start instantaneously, some ramp in
            # smoothly. Training on both prevents the model from learning
            # "sudden change" as a shortcut for "external cause".
            self._burst_ramp = bool(self.rng.random() < self.ramp_prob)
            self._burst_len = self.ramp_steps if self._burst_ramp else self.step_steps
            self._burst_left = self._burst_len
            self._burst_body = int(self.rng.choice(self.link_ids))
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-9
            magnitude = self.perturb_force * self.rng.uniform(0.5, 1.5)
            self._burst_force = np.zeros(6)
            self._burst_force[:3] = direction * magnitude
            self._onset = True

    def _apply_burst(self) -> bool:
        self.data.xfrc_applied[:] = 0.0
        if self._burst_left <= 0:
            return False

        if self._burst_ramp:
            # Hann envelope: smooth onset and offset, no discontinuity anywhere.
            # Controls for "is the model detecting external cause, or just
            # detecting a sudden change?"
            progress = 1.0 - (self._burst_left / self._burst_len)
            envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * progress))
        else:
            envelope = 1.0

        self.data.xfrc_applied[self._burst_body] = self._burst_force * envelope
        self._burst_left -= 1
        return True

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        """Advance one control step.

        Returns (next_obs, info). info carries the perturbation ground truth:
        `perturbed`, `onset` (first step of a burst), and `ramp` (smooth onset).
        All three are for MEASUREMENT ONLY and are never model inputs.
        """
        action = np.clip(action, -2.5, 2.5)
        self.data.ctrl[: self.nu] = action

        self._maybe_start_burst()
        perturbed = self._apply_burst()
        info = {"perturbed": perturbed, "onset": self._onset, "ramp": self._burst_ramp}

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Guard against blow-ups; a diverged sim poisons the buffer silently.
        if not np.all(np.isfinite(self.obs())):
            self.reset()

        return self.obs(), info
