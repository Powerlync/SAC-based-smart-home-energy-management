import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from smart_home_env import SmartHomeEnv
import os

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "helsinki.csv"
MODEL_PATH = "sac_.zip"
SEED = 42

N_EPISODES = 365
HOURS_PER_EPISODE = 24

assert os.path.exists(MODEL_PATH), f"❌ Model not found: {MODEL_PATH}"

# ============================================================
# LOAD ENV + MODEL
# ============================================================
env = Monitor(SmartHomeEnv(DATA_PATH, seed=SEED))
model = SAC.load(MODEL_PATH, env=env)

base_env = env.env  # unwrap Monitor

# ============================================================
# LOGGING
# ============================================================
t_hist, load_hist, pv_hist, chp_el_hist, grid_hist, bs_hist = [], [], [], [], [], []
soc_bs_hist, soc_ev_hist, T_in_hist, T_hw_hist, reward_hist = [], [], [], [], []

t = 0


# ============================================================
# RUN 365 EPISODES × 24 HOURS
# ============================================================
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    h = 0

    while not done and h < HOURS_PER_EPISODE:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        row = base_env.df.iloc[base_env.t]

        # Power flows
        P_bs = base_env.P_bs_max * action[0]
        P_ev = base_env.P_ev_max * max(action[1], 0.0) if row["EV_connected"] else 0.0
        P_hw = base_env.P_hw_max * max(action[2], 0.0)

        heat_deficit = max(0.0, base_env.T_target - base_env.T_in)
        P_chp_th = min(
            heat_deficit * base_env.C_air,
            base_env.P_chp_el_max * (base_env.eta_chp_th / base_env.eta_chp_el)
        )
        P_chp_el = P_chp_th * (base_env.eta_chp_el / base_env.eta_chp_th)

        P_demand = row["load_kw"] + P_hw + P_ev + max(P_bs, 0.0)
        P_supply = row["pv_kw"] + P_chp_el + max(-P_bs, 0.0)
        P_grid = P_demand - P_supply

        t_hist.append(t)
        load_hist.append(P_demand)
        pv_hist.append(row["pv_kw"])
        chp_el_hist.append(P_chp_el)
        grid_hist.append(P_grid)
        bs_hist.append(P_bs)

        soc_bs_hist.append(base_env.soc_bs)
        soc_ev_hist.append(base_env.soc_ev)
        T_in_hist.append(base_env.T_in)
        T_hw_hist.append(base_env.T_hw)
        reward_hist.append(reward)

        t += 1
        h += 1

# ============================================================
# Convert logs to DataFrame
# ============================================================
df = pd.DataFrame({
    "timestep": t_hist,
    "load": load_hist,
    "pv": pv_hist,
    "chp": chp_el_hist,
    "grid": grid_hist,
    "bs": bs_hist,
    "soc_bs": soc_bs_hist,
    "soc_ev": soc_ev_hist,
    "T_in": T_in_hist,
    "T_hw": T_hw_hist,
    "reward": reward_hist
})

# ============================================================
# YEARLY AVERAGES
# ============================================================
hours_per_year = 8760
df["year"] = df["timestep"] // hours_per_year
yearly_avg = df.groupby("year")[["load", "pv", "chp", "grid"]].mean()

# ============================================================
# GRADIENTS
# ============================================================
df["grad_load"] = np.gradient(df["load"])
df["grad_pv"] = np.gradient(df["pv"])
df["grad_chp"] = np.gradient(df["chp"])
df["grad_grid"] = np.gradient(df["grid"])

# ============================================================
# PLOTS
# ============================================================
plt.figure(figsize=(14, 5))
plt.plot(df["timestep"], df["load"], label="Total Demand", color="black")
plt.plot(df["timestep"], df["pv"], label="PV", color="gold")
plt.plot(df["timestep"], df["chp"], label="CHP Electricity", color="red")
plt.plot(df["timestep"], df["grid"], label="Grid (+import / -export)", color="blue")
plt.xlabel("Timestep (hour)")
plt.ylabel("Power (kW)")
plt.title("Electric Power Balance – Heat-led CHP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df["timestep"], df["soc_bs"], label="Battery SOC")
plt.plot(df["timestep"], df["soc_ev"], label="EV SOC")
plt.xlabel("Timestep")
plt.ylabel("SOC")
plt.title("State of Charge")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df["timestep"], df["T_in"], label="Indoor Temperature")
plt.axhline(base_env.T_in_min, linestyle="--", color="r", label="Comfort Min")
plt.axhline(base_env.T_in_max, linestyle="--", color="r", label="Comfort Max")
plt.xlabel("Timestep")
plt.ylabel("°C")
plt.title("Indoor Temperature")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df["timestep"], df["T_hw"], label="Hot Water Temperature", color="brown")
plt.axhline(base_env.T_hw_min, linestyle="--", color="r", label="HW Min")
plt.axhline(base_env.T_hw_max, linestyle="--", color="r", label="HW Max")
plt.xlabel("Timestep")
plt.ylabel("°C")
plt.title("Hot Water Tank Temperature")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df["timestep"], np.cumsum(df["reward"]), label="Cumulative Reward")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df["timestep"], df["grad_load"], label="Load Gradient")
plt.plot(df["timestep"], df["grad_pv"], label="PV Gradient")
plt.plot(df["timestep"], df["grad_chp"], label="CHP Gradient")
plt.plot(df["timestep"], df["grad_grid"], label="Grid Gradient")
plt.xlabel("Timestep (hour)")
plt.ylabel("Power Change (kW/hour)")
plt.title("Gradients of Energy Flows")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n===== Evaluation Summary =====")
print(f"Total reward: {df['reward'].sum():.2f}")
print(f"Final Battery SOC: {df['soc_bs'].iloc[-1]:.2f}")
print(f"Final EV SOC: {df['soc_ev'].iloc[-1]:.2f}")
print(f"Mean Indoor Temp: {df['T_in'].mean():.2f} °C")
print(f"Mean Hot Water Temp: {df['T_hw'].mean():.2f} °C")
print(f"Mean Grid Power: {df['grid'].mean():.2f} kW")

print("\nYearly Average Energy Flows (kW):")
print(yearly_avg)

print("\nGradient Statistics (kW/hour):")
print(df[["grad_load", "grad_pv", "grad_chp", "grad_grid"]].agg(['mean','std','min','max']))

grid_fraction = df["grid"].clip(lower=0).sum() / df["load"].sum()
pv_fraction = df["pv"].sum() / df["load"].sum()
chp_fraction = df["chp"].sum() / df["load"].sum()

print(f"\nFraction of Load Supplied by Grid (imports only): {grid_fraction*100:.2f}%")
print(f"Fraction of Load Supplied by PV: {pv_fraction*100:.2f}%")
print(f"Fraction of Load Supplied by CHP: {chp_fraction*100:.2f}%")

imbalance = np.max(
    np.abs(df["load"] - (df["pv"] + df["chp"] + np.maximum(-df["bs"], 0) + df["grid"]))
)
print(f"Max power imbalance: {imbalance:.6f} kW")

# ============================================================
# FULL-YEAR COST & CO₂ EMISSIONS (DO THIS FIRST)
# ============================================================
N_YEAR = len(df)

# Ensure price and carbon arrays match df length
price_buy = np.tile(base_env.df["price_eur_mwh"].values / 1000.0, N_YEAR // len(base_env.df) + 1)[:N_YEAR]  # €/kWh
price_sell = np.tile(base_env.df["price_sell_eur_mwh"].values / 1000.0, N_YEAR // len(base_env.df) + 1)[:N_YEAR]  # €/kWh
carbon_grid = np.tile(base_env.df["co2_g_kwh"].values / 1000.0, N_YEAR // len(base_env.df) + 1)[:N_YEAR]  # kg/kWh

# Grid import/export (FULL YEAR)
grid_import = df["grid"].clip(lower=0).values
grid_export = (-df["grid"].clip(upper=0)).values

# Grid electricity cost & emissions
grid_cost = grid_import * price_buy - grid_export * price_sell
grid_emissions = grid_import * carbon_grid

# CHP fuel, cost & emissions
chp_fuel = df["chp"].values / base_env.eta_chp_el
gas_price = 1.2  # €/kWh
chp_cost = chp_fuel * gas_price
carbon_gas = 0.202  # kg CO₂ per kWh of natural gas
chp_emissions = chp_fuel * carbon_gas


# Totals
total_cost = grid_cost.sum() + chp_cost.sum()
total_emissions = grid_emissions.sum() + chp_emissions.sum()

# ============================================================
# THERMAL COMFORT VIOLATIONS (FULL YEAR)
# ============================================================
df["T_in_violation"] = (df["T_in"] < base_env.T_in_min) | (df["T_in"] > base_env.T_in_max)
df["T_hw_violation"] = (df["T_hw"] < base_env.T_hw_min) | (df["T_hw"] > base_env.T_hw_max)

comfort_hours = int(df["T_in_violation"].sum())
hw_hours = int(df["T_hw_violation"].sum())
comfort_rate = comfort_hours / N_YEAR * 100

# ============================================================
# BATTERY USAGE & CONTROL SMOOTHNESS
# ============================================================
df["bs_ramp"] = np.abs(np.diff(df["bs"], prepend=0))
battery_throughput = np.abs(df["bs"]).sum()
avg_ramp_rate = df["bs_ramp"].mean()

# ============================================================
# HEURISTIC BASELINE (PV-FIRST, NO STORAGE, FULL YEAR)
# ============================================================
load = np.tile(base_env.df["load_kw"].values, N_YEAR // len(base_env.df) + 1)[:N_YEAR]
pv = np.tile(base_env.df["irradiance_kw_m2"].values, N_YEAR // len(base_env.df) + 1)[:N_YEAR]

grid_baseline = np.maximum(load - pv, 0)
baseline_cost = np.sum(grid_baseline * price_buy)
baseline_emissions = np.sum(grid_baseline * carbon_grid)

cost_saving = (baseline_cost - total_cost) / baseline_cost * 100
em_saving = (baseline_emissions - total_emissions) / baseline_emissions * 100

# ============================================================
# FINAL KPI SUMMARY (REPORT-READY)
# ============================================================
print("\n================ FINAL KPI SUMMARY ================")
print(f"Total Energy Cost (RL):        {total_cost:.2f} €")
print(f"Total Energy Cost (Baseline):  {baseline_cost:.2f} €")
print(f"Cost Difference:               {cost_saving:.2f} %")

print(f"\nTotal CO₂ Emissions (RL):       {total_emissions:.2f} kg")
print(f"Total CO₂ Emissions (Baseline): {baseline_emissions:.2f} kg")
print(f"Emission Reduction:            {em_saving:.2f} %")

print(f"\nIndoor Comfort Violations:      {comfort_hours} h ({comfort_rate:.2f}%)")
print(f"Hot Water Violations:          {hw_hours} h")

print(f"\nBattery Energy Throughput:      {battery_throughput:.2f} kWh")
print(f"Average Battery Ramp Rate:      {avg_ramp_rate:.3f} kW/h")
print("===================================================")
HOURS_PER_DAY = 24
df["day"] = df["timestep"] // HOURS_PER_DAY
daily_reward = df.groupby("day")["reward"].sum()
best_day = daily_reward.idxmax()
df_best = df[df["day"] == best_day]

print(f"\nBest day: Day {best_day} | Total reward = {daily_reward.loc[best_day]:.2f}")

# ============================================================
# THERMAL STATES – BEST DAY
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(df_best["timestep"], df_best["T_in"], label="Indoor Temperature", linewidth=2)
plt.plot(df_best["timestep"], df_best["T_hw"], label="Hot Water Temperature", linewidth=2)
plt.axhline(base_env.T_in_min, linestyle="--", alpha=0.6, label="Indoor Min")
plt.axhline(base_env.T_in_max, linestyle="--", alpha=0.6, label="Indoor Max")
plt.axhline(base_env.T_hw_min, linestyle="--", alpha=0.6, label="HW Min")
plt.axhline(base_env.T_hw_max, linestyle="--", alpha=0.6, label="HW Max")
plt.xlabel("Hour")
plt.ylabel("Temperature (°C)")
plt.title(f"Thermal States – Best Day {best_day}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
