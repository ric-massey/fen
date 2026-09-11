"""
Stage 2 environment: body, world, drives.

Observation is three streams, concatenated:

  proprioception  base position and velocity, arm joint angles and velocities,
                  tip position relative to the base
  exteroception   a smell-like gradient: egocentric direction to the nearest
                  charged food site plus an intensity that falls off with
                  distance. Chemotaxis, not vision — cheap and learnable.
  interoception   the fogged drive signal from drives.py

Note what is NOT sensed: ambient temperature. The agent feels its own body
temperature changing but never reads the thermal field directly, so it has to
infer where it is warm from what happens to it. That asymmetry is deliberate.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from drives import Drives, FoodSites
from fear import Fear
from hazard import Hazard, Weather
from health import Health
from reflex import Reflex

WORLD = Path(__file__).parent / "world.xml"


# ── FROZEN WORLD v1 ─────────────────────────────────────────────────────────
# The observation space and world parameters below are FROZEN. Every change to
# them invalidates the world model, every trained policy, and every measurement
# taken before the change — which is exactly what happened repeatedly while this
# was being built, leaving nothing comparable to anything else.
#
# Do not edit these to make a result better. Bump WORLD_VERSION, rerun the whole
# pipeline, and keep the old numbers labelled with the version they came from.
WORLD_VERSION = "v1"
OBS_DIM = 26
OBS_LAYOUT = {
    "base_xy": (0, 2), "base_vel": (2, 4), "arm_pos": (4, 8), "arm_vel": (8, 12),
    "tip_rel": (12, 14), "smell": (14, 17), "thermal_gradient": (17, 19),
    "hazard": (19, 22), "arousal": (22, 23), "drives": (23, 26),
}


class Stage2:
    def __init__(self, seed: int = 0, frame_skip: int = 5, fog: float = 1.0,
                 max_steps: int = 100_000, reflexes: bool = True,
                 temp_trigger: float = 0.55, energy_trigger: float = 0.30,
                 reflex_cost: float = 0.25, health_effects: bool = True,
                 randomize: bool = True, layout_seed: int | None = None,
                 danger: bool = True, weather: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(WORLD))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        # Layout randomisation gets its own stream so held-out worlds are
        # reproducible independently of the behaviour stream.
        self.randomize = randomize
        self.layout_rng = np.random.default_rng(
            seed if layout_seed is None else layout_seed)
        self.thermal_axis = np.array([1.0, 0.0], dtype=np.float32)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_steps = max_steps

        self.nu = self.model.nu                 # 6: vx, vy, 4 torques
        self.n_arm = 4
        self.tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")

        food_pos = np.array([[1.2, 1.0, 0.08], [2.1, -1.3, 0.08], [0.4, -2.0, 0.08]])
        self.food = FoodSites(food_pos)
        self.drives = Drives(seed=seed, dt=self.dt, fog=fog)
        self._base_fog = fog
        # Reflexes live in the body, not the learner. They are anatomy.
        self.reflex = (Reflex(temp_trigger=temp_trigger,
                              energy_trigger=energy_trigger) if reflexes else None)
        self.drives.reflex_cost = reflex_cost
        self.hazard = Hazard(seed=seed) if danger else None
        self.fear = Fear(dt=self.dt)
        self.weather = Weather(seed=seed) if weather else None
        self.health = Health(dt=self.dt,
                             effect_metabolism=health_effects,
                             effect_speed=health_effects,
                             effect_fog=health_effects)

        self.obs_dim = OBS_DIM
        self.world_version = WORLD_VERSION
        self.act_low = np.array([-1.2, -1.2, -2.5, -2.5, -2.5, -2.5], dtype=np.float32)
        self.act_high = -self.act_low

        o = self.reset()
        assert o.shape[0] == OBS_DIM, (
            f"observation is {o.shape[0]} but the frozen world declares {OBS_DIM}. "
            "Bump WORLD_VERSION and rerun the pipeline rather than editing OBS_DIM.")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _tip_xy(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.tip_sid][:2], dtype=np.float32)

    def _base_xy(self) -> np.ndarray:
        return np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float32)

    def _ambient(self) -> float:
        """Thermal field along a per-episode random axis. Never observed.

        The axis is randomised precisely to break the confound that made the
        old contingency gate meaningless: when warm was always +x, temperature
        was a function of x, so correlating temperature against x-velocity was
        correlating x against its own derivative. A blind sine wave passed it.
        With the axis re-drawn each life, no fixed spatial rule substitutes for
        actually sensing your own temperature.
        """
        proj = float(np.dot(self._base_xy(), self.thermal_axis))
        return float(np.clip(proj / 2.2, -1.0, 1.0))

    def _smell(self) -> np.ndarray:
        """Egocentric direction and intensity of the nearest charged food."""
        base = self._base_xy()
        charged = self.food.charge > 0.05
        if not charged.any():
            return np.zeros(3, dtype=np.float32)
        pos = self.food.pos[charged][:, :2]
        d = np.linalg.norm(pos - base[None, :], axis=1)
        i = int(np.argmin(d))
        direction = (pos[i] - base) / (d[i] + 1e-6)
        intensity = float(np.exp(-d[i] / 1.5))
        return np.array([direction[0], direction[1], intensity], dtype=np.float32)

    def _thermal_grad(self) -> np.ndarray:
        """Egocentric direction of increasing warmth, and its steepness.

        Exteroception, not interoception: this says which way is warmer, never
        whether the organism should go there. That decision requires reading its
        own temperature, so the two streams have to be combined. Clamping the
        internal state should flip the response to an identical gradient — which
        is exactly what the intervention test measures, and what no spatial rule
        or oscillation can fake.
        """
        g = self.thermal_axis.astype(np.float32)
        steep = float(np.clip(1.0 - abs(self._ambient()), 0.0, 1.0))
        return np.array([g[0], g[1] * 1.0, steep], dtype=np.float32)[:2]

    def obs(self) -> np.ndarray:
        base = self._base_xy()
        base_v = np.array([self.data.qvel[0], self.data.qvel[1]], dtype=np.float32)
        arm_q = np.asarray(self.data.qpos[2: 2 + self.n_arm], dtype=np.float32)
        arm_v = np.asarray(self.data.qvel[2: 2 + self.n_arm], dtype=np.float32)
        tip_rel = self._tip_xy() - base
        # Self-knowledge is not constant. It degrades with illness, and again
        # with fear — a frightened organism stops noticing it is hungry while
        # the threat itself becomes sharper. Attentional narrowing, as a
        # reallocation rather than extra information.
        self.drives.fog = (self._base_fog * self.health.fog_multiplier()
                           * self.fear.drive_fog_multiplier())
        hz = (self.hazard.sense(base) if self.hazard else np.zeros(3, dtype=np.float32))
        hz = hz * np.array([1.0, 1.0, self.fear.threat_sharpening()], dtype=np.float32)
        self.drives.health_sense = self.health.health
        return np.concatenate([
            base / 2.7, base_v, arm_q, arm_v, tip_rel,
            self._smell(), self._thermal_grad(),
            hz, np.array([self.fear.felt], dtype=np.float32),
            self.drives.sense(),
        ]).astype(np.float32)

    def true_log(self) -> dict:
        """Ground truth for measurement. Never an input."""
        e, t = self.drives.true_state()
        return {
            "energy": float(e), "temp": float(t),
            "x": float(self._base_xy()[0]), "y": float(self._base_xy()[1]),
            "deficit": float(self.drives.deficit(self.health.health)),
            "health": float(self.health.health),
            "stress": float(self.health.stress(self.drives.energy, self.drives.temp)),
            "food_charge": self.food.charge.copy(),
            "thermal_axis": self.thermal_axis.copy(),
            "hazard": (self.hazard.pos.copy() if self.hazard
                       else np.zeros(2, dtype=np.float32)),
            "hazard_radius": float(self.hazard.radius) if self.hazard else 0.0,
            "arousal": float(self.fear.arousal),
            "alive": bool(self.drives.alive),
        }

    def _sample_food(self) -> np.ndarray:
        """Food on BOTH sides of the thermal axis — a choice, not a compulsion.

        Biasing food toward the warm half forces the organism to overheat in
        order to eat. That is a compulsion, and it makes behaviour uninformative:
        there is only one thing to do.

        Guaranteeing at least one site on each side turns it into a valuation
        problem instead. The cool site is thermally safe; the warm site may be
        nearer, or richer by chance. Every episode the organism has to decide
        whether this food is worth the heat, and its answer is visible in what
        it does. That is a far better window on what it values than a forced
        trade-off, and it is what the intervention test can now actually read.
        """
        pts, want_warm = [], True
        guard = 0
        while len(pts) < 3 and guard < 500:
            guard += 1
            q = self.layout_rng.uniform(-2.3, 2.3, 2)
            if np.linalg.norm(q) < 0.8:
                continue
            side = float(np.dot(q, self.thermal_axis))
            if len(pts) < 2:
                # first two sites: one clearly warm, one clearly cool
                if want_warm and side < 0.6:
                    continue
                if not want_warm and side > -0.6:
                    continue
            if any(np.linalg.norm(q - e[:2]) < 1.0 for e in pts):
                continue
            pts.append(np.array([q[0], q[1], 0.08], dtype=np.float32))
            if len(pts) == 1:
                want_warm = False
        while len(pts) < 3:   # fallback if the guard tripped
            q = self.layout_rng.uniform(-2.3, 2.3, 2)
            pts.append(np.array([q[0], q[1], 0.08], dtype=np.float32))
        return np.array(pts, dtype=np.float32)

    # ── loop ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.rng.uniform(-1.0, 1.0)
        self.data.qpos[1] = self.rng.uniform(-1.0, 1.0)
        self.data.qpos[2: 2 + self.n_arm] = self.rng.uniform(-0.3, 0.3, self.n_arm)
        if self.randomize:
            ang = self.layout_rng.uniform(0, 2 * np.pi)
            self.thermal_axis = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            self.food.pos = self._sample_food()
            for i in range(len(self.food.pos)):
                self.data.mocap_pos[i] = self.food.pos[i]
        mujoco.mj_forward(self.model, self.data)
        self.food.reset()
        self.drives.reset()
        self.health.reset()
        self.fear.reset()
        if self.hazard:
            self.hazard.reset()
        if self.weather:
            self.weather.reset()
            self.thermal_axis = self.weather.step()
        self.steps = 0
        return self.obs()

    def step(self, action: np.ndarray):
        action = np.clip(action, self.act_low, self.act_high)

        # Reflexes read TRUE state and override the policy in emergencies.
        # The executed action is returned in info so the learner trains on
        # what actually happened, not on what it intended.
        reflex_fired = None
        if self.reflex is not None:
            e, t = self.drives.true_state()
            action, reflex_fired = self.reflex(action, float(e), float(t), self._smell())
            action = np.clip(action, self.act_low, self.act_high)

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if not np.all(np.isfinite(self.data.qpos)):
            self.reset()
            return self.obs(), -1.0, True, {"diverged": True}

        # The world moves whether or not the organism does. Nothing here is
        # inferable from what happened before.
        if self.weather:
            self.thermal_axis = self.weather.step()
        hazard_dmg = self.hazard.step(self._base_xy()) if self.hazard else 0.0
        if hazard_dmg:
            self.health.injure(hazard_dmg)
        prox = (float(self.hazard.sense(self._base_xy())[2]) if self.hazard else 0.0)
        self.fear.update(prox, hazard_dmg)

        prev_deficit = self.drives.deficit(self.health.health)
        ate = self.food.step(self._tip_xy())
        # Fear is metabolically expensive. Chronic arousal starves you, which
        # is what stops permanent panic being a free strategy.
        effort = (float(np.abs(action[2:]).sum())
                  * self.health.metabolism_multiplier()
                  * self.fear.metabolism_multiplier())
        self.drives.update(self._ambient(), effort, ate)

        # Neglect damages; damage persists and is felt. Healing costs energy.
        heal_spent = self.health.update(self.drives.energy, self.drives.temp)
        self.drives.energy = float(np.clip(self.drives.energy - heal_spent, 0.0, 1.0))
        self.drives.alive = not self.health.dead

        reward = self.drives.reward(prev_deficit, reflex_fired=reflex_fired is not None,
                                    health=self.health.health)

        self.steps += 1
        dead = not self.drives.alive
        # Development instances reset on death. The one-life rule applies to
        # Fen, not to dev-* runs (FEN.md, design principle 2).
        done = dead or self.steps >= self.max_steps

        info = self.true_log()
        cause = None
        if dead:
            cause = "hazard" if hazard_dmg > 0 else "neglect"
        info.update({"ate": float(ate), "dead": dead, "effort": effort,
                     "death_cause": cause, "reflex": reflex_fired,
                     "executed_action": action})
        return self.obs(), float(reward), done, info
