#!/bin/env python3

import argparse
import numpy as np
import os
import secrets
import functools
import datetime as dt
import matplotlib.pyplot as plt

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing, fft

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector.response import Response

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
from NuRadioReco.utilities.fft import freq2time, time2freq, freqs

import logging
logger = logging.getLogger("NuRadioMC.RNOG_emitter_simulation")
logger.setLevel(logging.INFO)

# Define all power string channels
power_string_channels = np.array([0, 1, 2, 3])

# Initialize modules for simulation
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(caching=False, pre_pulse_time=400 * units.ns)

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogHardwareResponse.begin(trigger_channels=power_string_channels)

highLowThreshold = highLowThreshold.triggerSimulator()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse()
rnogADCResponse.begin(clock_offset=0, adc_output="counts")


def detector_simulation(evt, station, det, noise_vrms, max_freq, add_noise=True):
    """ Run the detector simulation.

    It performs the following steps:
    - efieldToVoltageConverter: Convert the electric fields to voltages
    - channelGenericNoiseAdder: Add noise to the channels
    - rnogHardwareResponse: Apply the hardware response (for RADIANT and FLOWER channels)

    Parameters
    ----------
    evt : NuRadioMC.framework.event.Event
        The event to simulate the detector response for.
    station : NuRadioMC.framework.station.Station
        The station to simulate the detector response for.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    noise_vrms : float or dict of list
        The noise vrms (without any filter!). If a dict is given, the keys are the channel ids.
    max_freq : float
        The maximum frequency for the noise, i.e., the nyquist frequency for the simulated sampling rate.
    """

    efieldToVoltageConverter.run(evt, station, det, channel_ids=power_string_channels)
    if add_noise:
        channelGenericNoiseAdder.run(
            evt, station, det, amplitude=noise_vrms, min_freq=0 * units.MHz,
            max_freq=max_freq, type='rayleigh')

    rnogHardwareResponse.run(evt, station, det, sim_to_data=True)


def get_vrms_from_temperature_for_channels(det, station_id, channels, temperature):
    """ Get the vrms from the temperature for the channels. """
    vrms_per_channel = []
    for channel_id in channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=False)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(temperature=temperature, response=resp)
        )

    return np.array(vrms_per_channel)


def create_emitter_events(station_id, det, n_events=1, emitter_depth=-100*units.m, 
                          emitter_type="rnog_vpol", waveform_type="delta", 
                          amplitude=1.0, output_file=None):
    """
    Create emitter events for simulation.
    
    Parameters
    ----------
    station_id : int
        The station ID.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    n_events : int
        Number of events to generate.
    emitter_depth : float
        Depth of the emitter (negative is below surface).
    emitter_type : str
        Type of emitter ("rnog_vpol", "rnog_hpol", "delta", etc.).
    waveform_type : str
        Type of waveform to emit ("delta", "broadband", "mono", etc.).
    amplitude : float
        Amplitude of the emitter signal.
    output_file : str
        If provided, save the events to this file.
    
    Returns
    -------
    str or list
        Either the output file path if saved, or the list of events.
    """
    from NuRadioMC.SignalGen import emitter as emitter_signalgen
    from NuRadioReco.framework.base_trace import BaseTrace
    
    # Get station position
    station_pos = det.get_absolute_position(station_id)
    
    # Create list to store emitter events
    emitter_events = []
    
    for i in range(n_events):
        # Create emitter position (at the specified depth, but horizontally at the station position)
        emitter_pos = np.array([station_pos[0], station_pos[1], emitter_depth])
        
        # Set time for the event
        evt_time = dt.datetime(2023, 9, 9) + dt.timedelta(seconds=i)
        
        # Create vpol polarization (vertical)
        polarization = np.array([0, 0, 1])  # z-direction for vpol
        
        # Create the emitter
        emitter = emitter_signalgen.Emitter(position=emitter_pos,
                                           time=evt_time,
                                           name=emitter_type)
        
        # Set the emitter properties
        emitter.set_signal_type(waveform_type)
        emitter.set_amplitude(amplitude)
        emitter.set_polarization(polarization)
        
        # Append to list
        emitter_events.append(emitter)
    
    # Save to file if requested
    if output_file is not None:
        from NuRadioMC.utilities.hdf5_io import save_emitter_list
        save_emitter_list(output_file, emitter_events)
        return output_file
    
    return emitter_events


class EmitterSimulation(simulation.simulation):
    """
    Class to simulate emitter events (instead of neutrino events).
    """
    def __init__(self, *args, **kwargs):
        # Read config to get noise type
        tmp_config = simulation.get_config(kwargs["config_file"])
        
        def wrapper_detector_simulation(*args, **kwargs):
            noise_vrms = signal_processing.calculate_vrms_from_temperature(
                temperature=tmp_config['trigger']['noise_temperature'],
                bandwidth=tmp_config["sampling_rate"] / 2)

            detector_simulation(
                *args, **kwargs, noise_vrms=noise_vrms,
                max_freq=tmp_config["sampling_rate"] / 2)

        self._detector_simulation_part2 = wrapper_detector_simulation
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
    
    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
        rnogHardwareResponse.run(evt, station, det, sim_to_data=True)
        fig, ax = plt.subplots(2, 1, dpi=200)
        ax[0].set_xlabel("Time [ns]")
        ax[0].set_ylabel("Voltage [mV]")
        ax[1].set_xlabel("Frequency [GHz]")
        ax[1].set_ylabel("Amplitude [mV/GHz]")
        for ch in station.iter_channels():
            if ch.get_id() in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                # ax[1].set_xlim(0.175, 0.750)
                for i in range(2):
                    ax[i].grid()
                voltages = ch.get_trace()
                times = ch.get_times()
                vpkpk = (np.max(voltages) - np.min(voltages))/2
                # self.values[ch.get_id()].append(vpkpk)
                # h_int = get_hilbert_integral(voltages, times)
                ax[0].plot(times, voltages*1e3, label=f"ch{ch.get_id()}, vpk-pk={vpkpk*1e3:.2f} mV")
                # plt.xlim(900,1350)
                freq = freqs(len(times), sampling_rate = (times[1]-times[0])**-1)
                spec = time2freq(voltages*1e3, (times[1]-times[0])**-1) #units mV/GHz
                ax[1].plot(freq, np.abs(spec))
                ax[0].legend(fancybox=True, framealpha=0.5)
                plt.savefig(f"data/station_{args.station_id}/ch{ch.get_id()}/wf{ch.get_id()}.jpeg", bbox_inches='tight')
                ax[0].cla()
                ax[1].cla()

        
    def _detector_simulation_trigger(self, evt, station, det):
        # Empty implementation - we don't need trigger simulation for emitters
        # but the parent class will call this method
        pass


if __name__ == "__main__":
    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    default_config_path = os.path.join(ABS_PATH_HERE, "RNO_config.yaml")
    def_input_file = os.path.join(def_data_dir, f"input/station12/pulser_10.0.hdf5")

    parser = argparse.ArgumentParser(description="Run an RNO-G emitter simulation")

    # General steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to a NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None,
                        help="Path to a RNO-G detector description file. If None, query the description from hardware database")
    parser.add_argument("--station_id", type=int, default=11, help="Set station to be used in the simulation")
    
    # Emitter configuration
    parser.add_argument("--emitter_depth", type=float, default=-100, help="Depth of the emitter in meters (negative is below surface)")
    parser.add_argument("--emitter_type", type=str, default="RNOG_vpol_v3_5inch_center_n1.74", help="Type of emitter")
    parser.add_argument("--waveform", type=str, default="rno_cal5C_0dB", help="Type of waveform to emit")
    parser.add_argument("--amplitude", type=float, default=0.1, help="Amplitude of the emitter signal")
    parser.add_argument("--n_events", type=int, default=1, help="Number of events to simulate")
    
    # Input/Output configuration
    parser.add_argument("--emitter_file", type=str, default=def_input_file, help="Path to a file with pre-generated emitter events. If None, events will be generated on-the-fly.")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="Directory name where the output will be saved")
    parser.add_argument("--output", type=str, default=None, help="Output file name. If None, a default name will be generated.")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    args = parser.parse_args()
    
    # Initialize detector
    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id)

    event_time = dt.datetime(2023, 9, 9)
    det.update(event_time)
    config = simulation.get_config(args.config)
    
    # Create output directory
    output_path = os.path.join(args.data_dir, f"station_{args.station_id}/emitter_{args.emitter_type}")
    if not os.path.exists(output_path):
        logger.info(f"Create output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)
    
    # Generate default output filename if not provided
    if args.output is None:
        output_filename = os.path.join(output_path, 
                                       f"emitter_{args.emitter_type}_{args.waveform}_depth{abs(args.emitter_depth)}m.hdf5")
    else:
        output_filename = os.path.join(output_path, args.output)
    
    # Generate or load emitter events
    if args.emitter_file is None:
        # Generate events on the fly
        logger.info(f"Generating {args.n_events} emitter events")
        emitter_events = create_emitter_events(
            args.station_id, det, 
            n_events=args.n_events,
            emitter_depth=args.emitter_depth * units.m,
            emitter_type=args.emitter_type,
            waveform_type=args.waveform,
            amplitude=args.amplitude
        )
        input_data = emitter_events
    else:
        # Load events from file
        input_data = args.emitter_file
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input file {input_data} does not exist")
        logger.info(f"Reading emitter events from input file: {input_data}")
    
    # Set NUR output filename if requested
    if args.nur_output:
        nur_output_filename = output_filename.replace(".hdf5", ".nur")
    else:
        nur_output_filename = None
    
    # Initialize and run simulation
    sim = EmitterSimulation(
        inputfilename=input_data,
        outputfilename=output_filename,
        det=det,
        evt_time=event_time,
        outputfilenameNuRadioReco=nur_output_filename,
        config_file=args.config,
        use_cpp=True,
        debug=True,
        simulation_mode="emitter"  # Explicitly set emitter mode
    )
    
    # Run the simulation
    logger.info("Starting emitter simulation")
    sim.run()
    logger.info(f"Simulation completed. Output saved to {output_filename}")
    if args.nur_output:
        logger.info(f"NUR output saved to {nur_output_filename}")
