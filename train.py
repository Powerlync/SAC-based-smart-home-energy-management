import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from shenv import SmartHomeEnv  # heat-led CHP env


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "helsinki.csv"
TOTAL_TIMESTEPS = 500000
N_ENVS = 1
SEED = 42


# ============================================================
# ENV FACTORY
# =========================x===================================
def make_env():
    env = SmartHomeEnv(DATA_PATH, seed=SEED)
    return Monitor(env)


vec_env = make_vec_env(
    make_env,
    n_envs=N_ENVS,
    seed=SEED,
)


# ============================================================
# POLICY CONFIG
# ============================================================
policy_kwargs = dict(
    net_arch=[256, 256],
)


# ============================================================
# LOGGER
# ============================================================
logger = configure(
    folder="./sac_logs/",
    format_strings=["stdout", "csv"],
)


# ============================================================
# MODEL
# ============================================================
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    batch_size=256,
    buffer_size=1_000_000,
    learning_rate=3e-4,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=2,
    learning_starts=5_000,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
)

model.set_logger(logger)


# ============================================================
# CALLBACKS
# ============================================================
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="sac_shems_heat_ld_chp",
)

eval_env = Monitor(SmartHomeEnv(DATA_PATH, seed=SEED))

eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5_000,
    deterministic=True,
    render=False,
)


# ============================================================
# TRAIN
# ============================================================
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
)

model.save("sac_")
print("\n✅ Training completed. Model saved.\n")


# ============================================================
# EVALUATION ROLLOUT
# ============================================================
env = Monitor(SmartHomeEnv(DATA_PATH, seed=SEED))
obs, _ = env.reset()

done = False
total_reward = 0.0

load_trace = []
pv_trace = []
chp_trace = []
grid_trace = []
timesteps = []

t = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

    base_env = env.env  # unwrap Monitor
    row = base_env.df.iloc[base_env.t]

    # --------------------------------------------------------
    # Reconstruct power flows (EXACTLY as in environment)
    # --------------------------------------------------------
    P_bs = base_env.P_bs_max * action[0]
    P_ev = (
        base_env.P_ev_max * max(action[1], 0)
        if row["EV_connected"] else 0.0
    )
    P_hw = base_env.P_hw_max * max(action[2], 0)

    # Heat-led CHP (no action!)
    heat_deficit = max(0.0, base_env.T_target - base_env.T_in)
    P_chp_th = min(
        heat_deficit * base_env.C_air,
        base_env.P_chp_el_max * (base_env.eta_chp_th / base_env.eta_chp_el)
    )
    P_chp_el = P_chp_th * (base_env.eta_chp_el / base_env.eta_chp_th)

    P_demand = (
        row["load_kw"]
        + P_hw
        + P_ev
        + max(P_bs, 0)
    )

    P_supply = (
        row["pv_kw"]
        + P_chp_el
        + max(-P_bs, 0)
    )

    P_grid = P_demand - P_supply
    

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------
    load_trace.append(P_demand)
    pv_trace.append(row["pv_kw"])
    chp_trace.append(P_chp_el)
    grid_trace.append(P_grid)
    timesteps.append(t)

    t += 1


print(f"✅ Total evaluation reward: {total_reward:.2f}")



# ============================================================
# PLOTS
# ============================================================
