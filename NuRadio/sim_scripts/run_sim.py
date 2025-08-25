import argparse, copy, os, secrets, csv
import numpy as np
import datetime as dt
from scipy import constants
import matplotlib.pyplot as plt

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing
from NuRadioReco.utilities.fft import freq2time, time2freq, freqs
from NuRadioReco.detector.RNO_G import rnog_detector

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold

import logging
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)

import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.efieldToVoltageConverter
# import NuRadioReco.modules.trigger.simpleThreshold
# import NuRadioReco.modules.trigger.powerIntegration
# import NuRadioReco.modules.channelResampler
# import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=True)

# simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
# powerIntegration = NuRadioReco.modules.trigger.powerIntegration.triggerSimulator()
# channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
# channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
# electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()
# highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
# hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin(log_level=logging.WARNING)

# def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):

#     vrms_per_channel = []
#     for channel_id in trigger_channels:
#         resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
#         vrms_per_channel.append(
#             signal_processing.calculate_vrms_from_temperature(temperature=temperature, response=resp)
#         )

#     return vrms_per_channel

def RNO_G_HighLow_Thresh(lgRate_per_hz):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    # This applies for the VPol antennas
    # parameterization comes from Alan: https://radio.uchicago.edu/wiki/images/e/e6/2023.10.11_Simulating_RNO-G_Trigger.pdf
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0

class mySimulation(simulation.simulation):

    def __init__(self, *args, trigger_channel_noise_vrms=None, **kwargs):
        self.values = {}
        for ch in range(9):
            self.values[ch] = []
        logger.status(f'\n created values dict for channels')

        # this module is needed in super().__init__ to calculate the vrms
        self.rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
        self.rnogHardwareResponse.begin(trigger_channels=kwargs['trigger_channels'])

        super().__init__(*args, **kwargs)
        self.logger = logger
        self.deep_trigger_channels = kwargs['trigger_channels']

        self.highLowThreshold = highLowThreshold.triggerSimulator()
        self.rnogADCResponse = triggerBoardResponse.triggerBoardResponse()
        self.rnogADCResponse.begin(
            clock_offset=0.0, adc_output="counts")

        # future TODO: Add noise
        # self.channel_generic_noise_adder = channelGenericNoiseAdder.channelGenericNoiseAdder()
        # self.channel_generic_noise_adder.begin(seed=self._cfg['seed'])

        self.output_mode = {'Channels': self._config['output']['channel_traces'],
                            'ElectricFields': self._config['output']['electric_field_traces'],
                            'SimChannels': self._config['output']['sim_channel_traces'],
                            'SimElectricFields': self._config['output']['sim_electric_field_traces']}

        self.high_low_trigger_thresholds = {
            "10mHz": RNO_G_HighLow_Thresh(-2),
            "100mHz": RNO_G_HighLow_Thresh(-1),
            "1Hz": RNO_G_HighLow_Thresh(0),
            "3Hz": RNO_G_HighLow_Thresh(np.log10(3)),
        }
        assert trigger_channel_noise_vrms is not None, "Please provide the trigger channel noise vrms"
        self.trigger_channel_noise_vrms = trigger_channel_noise_vrms
        # self.trigger_channel_noise_vrms = 5*units.mV  # default value for testing

    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
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
                # h_int = get_hilbert_integral(voltages, times)
                ax[0].plot(times, voltages*1e3, label=f"ch{ch.get_id()}, vpk-pk={vpkpk*1e3:.2f} mV")
                # plt.xlim(900,1350)
                freq = freqs(len(times), sampling_rate = (times[1]-times[0])**-1)
                spec = time2freq(voltages*1e3, (times[1]-times[0])**-1) #units mV/GHz
                ax[1].plot(freq, np.abs(spec))
                ax[0].legend(fancybox=True, framealpha=0.5)
                plt.savefig(f"data/station_11/ch{ch.get_id()}/rx-pre-hard{ch.get_id()}.jpeg", bbox_inches='tight')
                ax[0].cla()
                ax[1].cla()

        logger.status(f'\t start applying hardware response for station {station.get_id()}, evt {evt.get_id()}')
        self.rnogHardwareResponse.run(evt, station, det, sim_to_data=True)
        logger.status(f'\t finished applying hardware response')

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
                self.values[ch.get_id()].append(vpkpk)
                logger.status(f'\n adding values for ch{ch.get_id()}')
                # h_int = get_hilbert_integral(voltages, times)
                ax[0].plot(times, voltages*1e3, label=f"ch{ch.get_id()}, vpk-pk={vpkpk*1e3:.2f} mV")
                # plt.xlim(900,1350)
                freq = freqs(len(times), sampling_rate = (times[1]-times[0])**-1)
                spec = time2freq(voltages*1e3, (times[1]-times[0])**-1) #units mV/GHz
                ax[1].plot(freq, np.abs(spec))
                ax[0].legend(fancybox=True, framealpha=0.5)
                plt.savefig(f"data/station_11/ch{ch.get_id()}/rx-post-hard{ch.get_id()}.jpeg", bbox_inches='tight')
                ax[0].cla()
                ax[1].cla()

    def return_values(self):
        return self.values

    def _detector_simulation_trigger(self, evt, station, det):
        logger.info('hi from trigger')
        # Runs the FLOWER board response
        # vrms_after_gain = self.rnogADCResponse.run(
        #     evt, station, det, trigger_channels=self.deep_trigger_channels,
        #     vrms=self.trigger_channel_noise_vrms, digitize_trace=True,
        # )

        # for idx, trigger_channel in enumerate(self.deep_trigger_channels):
        #     self.logger.debug(
        #         'Vrms = {:.2f} mV / {:.2f} mV (after gain).'.format(
        #             self.trigger_channel_noise_vrms[idx] / units.mV, vrms_after_gain[idx] / units.mV
        #         ))
        #     self._Vrms_per_trigger_channel[station.get_id()][trigger_channel] = vrms_after_gain[idx]

        # # this is only returning the correct value if digitize_trace=True for self.rnogADCResponse.run(..)
        # flower_sampling_rate = station.get_trigger_channel(self.deep_trigger_channels[0]).get_sampling_rate()
        # self.logger.debug('Flower sampling rate is {:.1f} MHz'.format(
        #     flower_sampling_rate / units.MHz
        # ))

        # for thresh_key, threshold in self.high_low_trigger_thresholds.items():

        #     if self.rnogADCResponse.adc_output == "voltage":
        #         threshold_high = {channel_id: threshold * vrms for channel_id, vrms
        #             in zip(self.deep_trigger_channels, vrms_after_gain)}
        #         threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms
        #             in zip(self.deep_trigger_channels, vrms_after_gain)}
        #     else:
        #         # We round here. This is not how an ADC works but I think this is not needed here.
        #         threshold_high = {channel_id: int(round(threshold * vrms)) for channel_id, vrms
        #             in zip(self.deep_trigger_channels, vrms_after_gain)}
        #         threshold_low = {channel_id: int(round(-1 * threshold * vrms)) for channel_id, vrms
        #             in zip(self.deep_trigger_channels, vrms_after_gain)}

        #     self.highLowThreshold.run(
        #         evt, station, det,
        #         threshold_high=threshold_high,
        #         threshold_low=threshold_low,
        #         use_digitization=False, #the trace has already been digitized with the rnogADCResponse
        #         high_low_window=6 / flower_sampling_rate,
        #         coinc_window=20 / flower_sampling_rate,
        #         number_concidences=2,
        #         triggered_channels=self.deep_trigger_channels,
        #         trigger_name=f"deep_high_low_{thresh_key}",
        #     )


if __name__ == "__main__":
    logger.status(f"\n beginning RNO-G pulser simulation")

    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    default_config_path = os.path.join(ABS_PATH_HERE, "RNO_config.yaml")

    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    # Sim steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None, help="Path to RNO-G detector description file. If None, query from DB")
    parser.add_argument("--station_id", type=int, default=11, help="Set station to be used for simulation", required=True)

    # # Neutrino arguments
    # parser.add_argument("--energy", '-e', default=1e18, type=float, help="Neutrino energy [eV]")
    # parser.add_argument("--flavor", '-f', default="all", type=str, help="the flavor")
    # parser.add_argument("--interaction_type", '-it', default="ccnc", type=str, help="interaction type cc, nc or ccnc")

    # File meta-variables
    parser.add_argument("--index", '-i', default=0, type=int, help="counter to create a unique data-set identifier")
    parser.add_argument("--n_events_pser_file", '-n', type=int, default=1, help="Number of nu-interactions per file")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="directory name where the library will be created")
    parser.add_argument("--proposal", action="store_true", help="Use PROPOSAL for simulation")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    args = parser.parse_args()
    kwargs = args.__dict__
    assert args.station_id is not None, "Please specify a station id with `--station_id`"

    root_seed = secrets.randbits(128)
    # deep_trigger_channels = np.array([0, 1, 2, 3])
    deep_trigger_channels = None 

    logger.status(f"\n loading detector description")
    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id)
    logger.status(f"\n finished loading detector description")

    logger.status(f'\t Detector updated to {dt.datetime(2022, 9, 9)}')
    det.update(dt.datetime(2022, 9, 9))

    logger.status(f"\t loading config")
    config = simulation.get_config(args.config)
    logger.status(f"\t finished loading config")

    # Get the trigger channel noise vrms
    # trigger_channel_noise_vrms = get_vrms_from_temperature_for_trigger_channels(
    #     det, args.station_id, deep_trigger_channels, config['trigger']['noise_temperature'])
    trigger_channel_noise_vrms = 0.001 * units.V  # default value for testing

    # # Simulate fiducial volume around station
    # volume = get_fiducial_volume(args.energy)
    # pos = det.get_absolute_position(args.station_id)
    # logger.info(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")
    # volume.update({"x0": pos[0], "y0": pos[1]})

    output_path = f"{args.data_dir}/station_{args.station_id}"

    if not os.path.exists(output_path):
        logger.info("Making dirs", output_path)
        os.makedirs(output_path, exist_ok=True)

    output_filename = (f"{output_path}/output.hdf5")

    if args.nur_output:
        nur_output_filename = output_filename.replace(".hdf5", ".nur")
    else:
        nur_output_filename = None

    logger.status(f'\n initialize simulation')
    sim = mySimulation(
        inputfilename=os.path.join(def_data_dir, "input/station11/pulser_10.0.hdf5"),
        outputfilename=output_filename,
        det=det,
        evt_time=dt.datetime(2022, 9, 9),
        outputfilenameNuRadioReco=nur_output_filename,
        config_file=args.config,
        trigger_channels=deep_trigger_channels,
        trigger_channel_noise_vrms=trigger_channel_noise_vrms,
        debug=True
    )
    logger.status(f'\n finished initializing simulation')

    logger.status(f"\n begin sim.run()")
    sim.run()
    logger.status(f"\n finished sim.run()")
    
    # Retrieve values and save to CSV
    logger.status(f"\t retrieving simulation values")
    values = sim.return_values()
    csv_output_path = os.path.join(output_path, "sim_values.csv")
    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Channel ID", "Values"])  # Header row
        for channel_id, channel_values in values.items():
            writer.writerow([channel_id, ','.join(map(str, channel_values))])

    logger.status(f"\t simulation values saved to {csv_output_path}")