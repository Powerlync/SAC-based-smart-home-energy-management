# SAC-Based Smart Home Energy Management

This project implements a **reinforcement learning–based energy management system** for a residential **multi-energy microgrid** using the **Soft Actor-Critic (SAC)** algorithm.

The controller coordinates photovoltaic (PV) generation, battery energy storage, electric vehicle (EV) charging, domestic hot water (DHW) heating, indoor thermal dynamics, and a combined heat and power (CHP) unit to minimize **electricity cost and carbon emissions** while maintaining **occupant comfort**.

## Key Features
- Model-free control using **Soft Actor-Critic (SAC)**
- Joint optimization of **electrical and thermal subsystems**
- **Carbon-aware** and cost-aware reward design
- Continuous control of battery, EV charging, and DHW heating
- Physically realistic constraints (e.g. PV-only battery charging)
- Hourly simulation with realistic synthetic data (Helsinki climate)

## System Components
- Rooftop PV generation  
- Battery Energy Storage System (BESS)  
- Electric Vehicle (EV) with availability constraints  
- Domestic Hot Water (DHW) system  
- Combined Heat and Power (CHP) unit  
- Grid interaction with dynamic prices and carbon intensity  

## Methodology
- Custom **Gymnasium** environment for residential multi-energy microgrids
- Continuous action space with embedded physical constraints
- Multi-objective reward balancing cost, emissions, comfort, and asset health
- Off-policy training with experience replay and entropy regularization

## Results (Summary)
- ~20% reduction in electricity cost  
- ~24% reduction in CO₂ emissions  
- Indoor and hot water temperatures maintained within comfort ranges  
- Smooth and physically consistent control actions  


## Requirements
- Python 3.9+
- Gymnasium
- Stable-Baselines3
- NumPy, Pandas, Matplotlib

## Author
**Sreehari Ramachandran**  
Aalto University, 2026

