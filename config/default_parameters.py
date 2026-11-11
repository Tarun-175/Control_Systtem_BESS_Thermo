"""Illustrative four-zone model parameters; not validated installation data."""
import numpy as np
CAPACITANCE = np.array([2.2e5, 2.0e5, 2.1e5, 2.3e5])  # J/K
NATURAL_H = np.array([420., 400., 410., 430.])       # W/K
COOLING_BETA = np.array([1800., 1750., 1850., 1800.]) # W/K at full command
COUPLING = np.array([[0,70,65,0],[70,0,0,65],[65,0,0,70],[0,65,70,0.]]) # W/K
GAMMA = np.array([0.010,0.0105,0.0098,0.0102])       # W/A^2 illustrative
DEFAULTS = dict(ambient=298.15, current=180., setpoint=303.15, duration=1800., dt=2., budget=2.0)
