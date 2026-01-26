import gymnasium as gym
import numpy as np
import pandas as pd


class SmartHomeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path, seed=None):
        super().__init__()

        self.df_base = pd.read_csv(
            data_path,
            parse_dates=["startTime"]
        ).fillna(0.0)

        self.rng = np.random.default_rng(seed)

        # --------------------------------------------------
        # EPISODE
        # --------------------------------------------------
        self.dt = 1.0
        self.episode_len = 24

        # --------------------------------------------------
        # REQUIRED COLUMNS (MATCH YOUR DATA)
        # --------------------------------------------------
        required = [
            "price_eur_mwh", "price_sell_eur_mwh",
            "co2_g_kwh", "load_kw",
            "irradiance_kw_m2",
            "T_out_C", "T_in_C",
            "hot_water_draw",
            "EV_connected",
            "hour"
        ]
        for c in required:
            if c not in self.df_base.columns:
                raise ValueError(f"Missing column: {c}")

        # --------------------------------------------------
        # PV MODEL
        # --------------------------------------------------
        PV_AREA = 25.0
        PV_EFF = 0.18
        PV_MAX = 5.0

        self.df_base["pv_kw"] = (
            self.df_base["irradiance_kw_m2"] * PV_AREA * PV_EFF
        ).clip(0, PV_MAX)

        # --------------------------------------------------
        # BATTERY
        # --------------------------------------------------
        self.E_bs = 13.5      # kWh
        self.P_bs_max = 3.0  # kW
        self.eta_bs = 0.95

        # --------------------------------------------------
        # EV
        # --------------------------------------------------
        self.E_ev = 40.0
        self.P_ev_max = 7.0
        self.eta_ev = 0.95
        self.ev_soc_req = 0.8

        # --------------------------------------------------
        # THERMAL BUILDING
        # --------------------------------------------------
        self.C_air = 3.0
        self.k_loss = 0.05
        self.T_in_min = 19.0
        self.T_in_max = 24.0
        self.T_target = 21.0

        # --------------------------------------------------
        # HOT WATER
        # --------------------------------------------------
        self.P_hw_max = 3.0
        self.C_hw = 2.0
        self.T_hw_min = 45.0
        self.T_hw_max = 60.0

        # --------------------------------------------------
        # CHP (HEAT-LED)
        # --------------------------------------------------
        self.P_chp_el_max = 5.0
        self.eta_chp_el = 0.30
        self.eta_chp_th = 0.55
        self.chp_fuel_cost = 0.12     # €/kWh_fuel
        self.chp_co2_factor = 0.25    # kgCO2/kWh_fuel

        # --------------------------------------------------
        # SPACES
        # --------------------------------------------------
        # Actions:
        # 0: Battery (- discharge, + charge)
        # 1: EV charging
        # 2: Hot water heater
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        # Observations
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(13,), dtype=np.float32
        )

        # Normalization
        self.max_load = self.df_base["load_kw"].max()
        self.max_pv = max(self.df_base["pv_kw"].max(), 1e-3)
        self.max_price = max(
            self.df_base["price_eur_mwh"].max(),
            self.df_base["price_sell_eur_mwh"].max()
        )
        self.max_co2 = self.df_base["co2_g_kwh"].max()

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
        # HEAT-LED CHP (MANDATORY)
        # --------------------------------------------------
        heat_deficit = max(0.0, self.T_target - self.T_in)

        P_chp_th = min(
            heat_deficit * self.C_air,
            self.P_chp_el_max * (self.eta_chp_th / self.eta_chp_el)
        )

        P_chp_el = P_chp_th * (self.eta_chp_el / self.eta_chp_th)
        fuel_chp = P_chp_el / self.eta_chp_el

        # Indoor temperature update
        self.T_in += (
            P_chp_th / self.C_air
            - self.k_loss * (self.T_in - row["T_out_C"])
        )

        # --------------------------------------------------
        # HOT WATER
        # --------------------------------------------------
        self.T_hw += P_hw / self.C_hw - 0.02 * row["hot_water_draw"]
        self.T_hw = np.clip(self.T_hw, 20, 80)

        # --------------------------------------------------
        # BATTERY
        # --------------------------------------------------
        soc_prev = self.soc_bs

        if P_bs_cmd >= 0:
            P_bs = min(
                P_bs_cmd,
                (1 - self.soc_bs) * self.E_bs / self.eta_bs
            )
            self.soc_bs += (P_bs * self.eta_bs) / self.E_bs
        else:
            P_bs = max(
                P_bs_cmd,
                -self.soc_bs * self.E_bs * self.eta_bs
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
        # ELECTRIC POWER BALANCE
        # --------------------------------------------------
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

        # --------------------------------------------------
        # COSTS
        # --------------------------------------------------
        buy = row["price_eur_mwh"] / 1000
        sell = row["price_sell_eur_mwh"] / 1000

        C_grid = buy * max(P_grid, 0) - sell * max(-P_grid, 0)
        C_grid_co2 = (row["co2_g_kwh"] / 1000) * max(P_grid, 0)

        C_chp_cost = self.chp_fuel_cost * fuel_chp
        C_chp_co2 = self.chp_co2_factor * fuel_chp

        # --------------------------------------------------
        # COMFORT PENALTY
        # --------------------------------------------------
        viol_T = max(self.T_in_min - self.T_in, 0) + max(self.T_in - self.T_in_max, 0)
        viol_HW = max(self.T_hw_min - self.T_hw, 0) + max(self.T_hw - self.T_hw_max, 0)

        C_comfort = viol_T**2 + 0.5 * viol_HW**2

        # --------------------------------------------------
        # REWARD
        # --------------------------------------------------
        C_smooth = 0.05 * np.sum(np.abs(a - self.prev_action))
        self.prev_action = a.copy()

        reward = (
            - C_grid
            - C_grid_co2
            - C_chp_cost
            - C_chp_co2
            - 2.0 * C_comfort
            - 0.1 * abs(self.soc_bs - soc_prev)
            - C_smooth
        )

        # --------------------------------------------------
        # TERMINATION
        # --------------------------------------------------
        self.t += 1
        terminated = self.t >= self.episode_len - 1
        truncated = False

        if terminated and self.soc_ev < self.ev_soc_req:
            reward -= 10.0 * (self.ev_soc_req - self.soc_ev)

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