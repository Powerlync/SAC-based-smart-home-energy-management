import gymnasium as gym
import numpy as np
import pandas as pd


class SmartHomeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path, seed=None):
        super().__init__()

        # ====================================================
        # LOAD DATA
        # ====================================================
        self.df_base = pd.read_csv(
            data_path,
            parse_dates=["startTime"]
        ).fillna(0.0)

        self.rng = np.random.default_rng(seed)

        # ====================================================
        # TIME
        # ====================================================
        self.dt = 1.0
        self.episode_len = 24

        # ====================================================
        # PV MODEL (MIDDAY PEAK, NO NIGHT PV)
        # ====================================================
        # ====================================================
# PV MODEL (FROM IRRADIANCE)
# ====================================================
 


        PV_AREA = 20.0 # m²
        PV_EFF = 0.18 # panel efficiency
        PV_MAX = 6.0 # inverter limit (kW)


        # Parameters for Gaussian noise: mean=0.55 (middle of 0.2-0.9), std=0.15
        mean_val = 0.55
        std_val = 0.15


        # Generate Gaussian noise for each row
        noise = np.random.normal(loc=mean_val, scale=std_val, size=len(self.df_base))


        # Clip noise to stay within 0.2 - 0.9
        noise = np.clip(noise, 0, 0.8)


        # Calculate pv_kw with random base instead of fixed 0.9
        self.df_base["pv_kw"] = (noise +
        self.df_base["irradiance_kw_m2"] * PV_AREA * PV_EFF * 2
        )


        # Clip to PV_MAX
        self.df_base["pv_kw"] = np.clip(self.df_base["pv_kw"], 0, PV_MAX)
 
      # ====================================================
        # NORMALIZATION CONSTANTS
        # ====================================================
        self.max_load = max(self.df_base["load_kw"].max(), 0.1)
        self.max_pv = max(self.df_base["pv_kw"].max(), 0.1)
        self.max_price = max(
            self.df_base["price_eur_mwh"].max(),
            self.df_base["price_sell_eur_mwh"].max(),
            1.0
        )
        self.max_co2 = max(self.df_base["co2_g_kwh"].max(), 1.0)

        # ====================================================
        # BATTERY (BESS)
        # ====================================================
        self.E_bs = 13.5
        self.P_bs_max = 3.0
        self.eta_bs = 0.95

        # ====================================================
        # EV
        # ====================================================
        self.E_ev = 40.0
        self.P_ev_max = 7.0
        self.eta_ev = 0.95
        self.ev_soc_req = 0.9

        # ====================================================
        # THERMAL BUILDING
        # ====================================================
        self.C_air = 3.0
        self.k_loss = 0.05
        self.T_in_min = 19.0
        self.T_in_max = 24.0
        self.T_target = 21.0

        # ====================================================
        # HOT WATER
        # ====================================================
        self.P_hw_max = 3.0
        self.C_hw = 2.0
        self.T_hw_min = 45.0
        self.T_hw_max = 60.0

        # ====================================================
        # CHP
        # ====================================================
        self.P_chp_el_max = 5.0
        self.eta_chp_el = 0.30
        self.eta_chp_th = 0.55
        self.chp_fuel_cost = 0.12
        self.chp_co2_factor = 0.25

        # ====================================================
        # ACTIONS
        # ====================================================
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        # ====================================================
        # OBSERVATIONS
        # ====================================================
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(13,), dtype=np.float32
        )

    # ====================================================
    # RESET
    # ====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start = self.rng.integers(
            0, len(self.df_base) - self.episode_len
        )

        self.df = self.df_base.iloc[
            start:start + self.episode_len
        ].reset_index(drop=True)

        self.t = 0
        self.soc_bs = 0.5
        self.soc_ev = self.rng.uniform(0.3, 0.6)
        self.T_in = self.df.loc[0, "T_in_C"]
        self.T_hw = 50.0
        self.prev_action = np.zeros(3, dtype=np.float32)

        return self._get_obs(), {}

    # ====================================================
    # STEP
    # ====================================================
    def step(self, action):
        a = np.clip(action, -1, 1)
        row = self.df.iloc[self.t]

        # --------------------------------------------------
        # ACTIONS
        # --------------------------------------------------
        P_bs_cmd = self.P_bs_max * a[0]
        P_ev = self.P_ev_max * max(a[1], 0) if row["EV_connected"] else 0.0
        P_hw = self.P_hw_max * max(a[2], 0)

        # --------------------------------------------------
        # CHP (HEAT-LED, PV-AWARE)
        # --------------------------------------------------
        heat_deficit = max(0.0, self.T_target - self.T_in)
        pv_factor = max(0.0, 1.0 - row["pv_kw"] / self.P_chp_el_max)

        P_chp_th = min(
            heat_deficit * self.C_air * pv_factor,
            self.P_chp_el_max * (self.eta_chp_th / self.eta_chp_el)
        )

        P_chp_el = P_chp_th * (self.eta_chp_el / self.eta_chp_th)
        fuel_chp = P_chp_el * self.dt / self.eta_chp_el

        self.T_in += (
            P_chp_th / self.C_air
            - self.k_loss * (self.T_in - row["T_out_C"])
        )

        # --------------------------------------------------
        # HOT WATER
        # --------------------------------------------------
        self.T_hw += (
            P_hw * self.dt / self.C_hw
            - 0.02 * row["hot_water_draw"]
        )
        self.T_hw = np.clip(self.T_hw, 20, 80)

        # --------------------------------------------------
        # BATTERY (PV-ONLY CHARGING)
        # --------------------------------------------------
        soc_prev = self.soc_bs

        P_net_load = row["load_kw"] + P_hw + P_ev
        P_local_gen = row["pv_kw"] + P_chp_el
        pv_surplus = max(P_local_gen - P_net_load, 0.0)

        if P_bs_cmd > 0:
            P_bs = min(
                P_bs_cmd,
                pv_surplus,
                (1 - self.soc_bs) * self.E_bs
            )
            self.soc_bs += (P_bs * self.eta_bs) / self.E_bs
        else:
            P_bs = max(
                P_bs_cmd,
                -self.soc_bs * self.E_bs
            )
            self.soc_bs += (P_bs / self.eta_bs) / self.E_bs

        self.soc_bs = np.clip(self.soc_bs, 0, 1)

        # --------------------------------------------------
        # EV
        # --------------------------------------------------
        if row["EV_connected"]:
            self.soc_ev += (P_ev * self.eta_ev) / self.E_ev
            self.soc_ev = np.clip(self.soc_ev, 0, 1)

        # --------------------------------------------------
        # POWER BALANCE
        # --------------------------------------------------
        P_demand = row["load_kw"] + P_hw + P_ev
        P_supply = row["pv_kw"] + P_chp_el + max(-P_bs, 0)

        P_grid = P_demand - P_supply
        E_grid = P_grid * self.dt

        # --------------------------------------------------
        # COSTS
        # --------------------------------------------------
        buy = row["price_eur_mwh"] / 1000
        sell = row["price_sell_eur_mwh"] / 1000

        C_grid = buy * max(E_grid, 0) - sell * max(-E_grid, 0)
        C_grid_co2 = (row["co2_g_kwh"] / 1000) * max(E_grid, 0)

        C_chp_cost = self.chp_fuel_cost * fuel_chp
        C_chp_co2 = self.chp_co2_factor * fuel_chp

        # --------------------------------------------------
        # PV UTILIZATION
        # --------------------------------------------------
        pv_used = min(P_local_gen, P_demand + max(P_bs, 0))
        pv_curtailed = max(row["pv_kw"] - pv_used, 0)

        R_pv = 0.05 * pv_used * self.dt
        C_curtail = 0.1 * pv_curtailed * self.dt

        # --------------------------------------------------
        # COMFORT
        # --------------------------------------------------
        viol_T = max(self.T_in_min - self.T_in, 0) + max(self.T_in - self.T_in_max, 0)
        viol_HW = max(self.T_hw_min - self.T_hw, 0) + max(self.T_hw - self.T_hw_max, 0)
        C_comfort = viol_T**2 + 0.5 * viol_HW**2

        # --------------------------------------------------
        # BATTERY DEGRADATION
        # --------------------------------------------------
        C_deg = 0.02 * abs(self.soc_bs - soc_prev)

        # --------------------------------------------------
        # REWARD
        # --------------------------------------------------
        C_smooth = 0.05 * np.sum(np.abs(a - self.prev_action))
        self.prev_action = a.copy()

        reward = (
            -C_grid
            -C_grid_co2
            -C_chp_cost
            -C_chp_co2
            -2.0 * C_comfort
            -0.3 * (self.soc_bs - 0.5) ** 2
            -C_deg
            -C_smooth
            -C_curtail
            + R_pv
        )

        # --------------------------------------------------
        # TERMINATION
        # --------------------------------------------------
        self.t += 1
        terminated = self.t >= self.episode_len - 1
        truncated = False

        if terminated:
            reward -= 20.0 * abs(self.soc_bs - 0.5)
            if self.soc_ev < self.ev_soc_req:
                reward -= 30.0 * (self.ev_soc_req - self.soc_ev)

        return self._get_obs(), reward, terminated, truncated, {}

    # ====================================================
    # OBSERVATION
    # ====================================================
    def _get_obs(self):
        row = self.df.iloc[min(self.t, self.episode_len - 1)]

        obs = np.array([
            self.soc_bs,
            self.soc_ev,
            self.T_in / 30.0,
            self.T_hw / 70.0,
            row["pv_kw"] / self.max_pv,
            row["load_kw"] / self.max_load,
            row["T_out_C"] / 35.0,
            row["price_eur_mwh"] / self.max_price,
            row["price_sell_eur_mwh"] / self.max_price,
            row["co2_g_kwh"] / self.max_co2,
            row["EV_connected"],
            row["hour"] / 23.0,
            (self.episode_len - self.t) / self.episode_len
        ], dtype=np.float32)

        return obs * 2 - 1