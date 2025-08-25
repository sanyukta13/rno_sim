from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import write_events_to_hdf5
import numpy as np
import os
import logging
import argparse
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector import detector
import datetime as dt


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('station_id', type=int, default=11, help='station id')
args = parser.parse_args()

if not os.path.exists(f"data/input/station{args.station_id}"):
    os.makedirs(f"data/input/station{args.station_id}")


if args.station_id != 0:
    det = rnog_detector.Detector(
    detector_file=None, log_level=logging.INFO,
    always_query_entire_description=False, select_stations=args.station_id)

    det.update(dt.datetime(2023, 9, 9))
else:
    det_file="RNO_station11.json"
    det=detector.Detector(det_file)
    det.update(dt.datetime(2023, 9, 9))


pos = det.get_absolute_position(args.station_id)
try:
    rel0=det.get_channel(args.station_id,0)['channel_position']['position']
    rel3=det.get_channel(args.station_id,3)['channel_position']['position']
except:
    rel0=[0,0,det.get_channel(args.station_id,0)["ant_position_z"]]
    rel3=[0,0,det.get_channel(args.station_id,3)["ant_position_z"]]

print(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m, z0={pos[2]:.2f}m")
print(f"PA channel x={rel0[0]:.2f}m, y={rel0[1]:.2f}m, z={rel0[2]:.2f}m")
print(f"Zshifted PA channel x={rel0[0]:.2f}m, y={rel0[1]:.2f}m, z={rel0[2]+pos[2]:.2f}m")

n_events = 1
obs_angles = np.linspace(-60., 60., 121) * units.deg


# Elevation of the vertex point seen from the middle point of the array
theta_file=[]
for obs_angle in obs_angles:
    print(obs_angle)
    file=f"pulser_{obs_angle/units.deg:.1f}"
    filename=f"data/input/station{args.station_id}/{file}.hdf5"

    # first set the meta attributes
    attributes = {}
    n_events = int(n_events)
    attributes['simulation_mode'] = "emitter"  # must be specified to work for radio emittimg models 
    attributes['n_events'] = n_events  # the number of events contained in this file
    attributes['start_event_id'] = 0  

    ########### FOR Emitter ###############
    data_sets = {}
    data_sets["emitter_antenna_type"] = ["RNOG_vpol_v3_5inch_center_n1.74"] * n_events
    data_sets["emitter_model"] = ["rno_cal5C_0dB"] * n_events #rnog_pulser
    data_sets["emitter_amplitudes"] = np.ones(n_events) * 0.1 * units.V # amplitude of the emitter antenna

    data_sets["xx"] = np.ones(n_events)* (pos[0]+rel0[0]) * units.m #102 
    data_sets["yy"] = np.ones(n_events) * (pos[1]+rel0[1]+34.5) * units.m #2950
    data_sets["zz"] = np.ones(n_events) * (pos[2]+34.5*np.tan(obs_angle)+rel0[2]) * units.m

    print(obs_angle/units.deg,data_sets["xx"][0],data_sets["yy"][0],data_sets["zz"][0])

    # the orientation of the emiting antenna, defined via two vectors that are defined with two angles each (see https://nu-radio.github.io/NuRadioReco/pages/detector_database_fields.html)
    # the following definition specifies a traditional “upright” dipole.
    data_sets["emitter_orientation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_orientation_theta"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_theta"] = np.ones(n_events) * 90 * units.deg

    # the following informations not particularly useful for radio_emitter models

    data_sets["event_group_ids"] = np.arange(n_events)
    data_sets["shower_ids"] = np.arange(n_events)

    # write events to file
    write_events_to_hdf5(filename, data_sets, attributes)

if not os.path.exists(f"data/output/station{args.station_id}"):
    os.mkdir(f"data/output/station{args.station_id}")

if not os.path.exists(f"data/output/station{args.station_id}/pulser"):
    os.mkdir(f"data/output/station{args.station_id}/pulser")

np.savetxt(f"data/output/station{args.station_id}/pulser/pulser_events.txt",obs_angles/units.deg,fmt="%.1f")
