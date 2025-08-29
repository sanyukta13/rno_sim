# contains modules required for custom_sim.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def read_hdf5(file):
    """
    Reads an HDF5 file and returns the voltage and time data.
    -----
    input: file (str): Path to the HDF5 file.
    -----
    Returns: wf: Voltage data.
             time: Time data.
    """
    with h5py.File(file, 'r') as f:
        wf = f['voltage'][:]
        time = f['time'][:]
    return wf, time

# def read_antmodel(file):
