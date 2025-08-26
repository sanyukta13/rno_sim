import os, glob, scipy, math, sys
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

sys.path.append(os.path.abspath('/home/sany/cathSim/'))
# from reading.data_reading import *
# from functions.functions import *
from IceCube_gen2_radio.IC_hybrid_station import *
from IceCube_gen2_radio.rno_g_st import rno_g_st
from IceCube_gen2_radio.tools import *
from IceCube_gen2_radio.tools_rno import *
from IceCube_gen2_radio.CATH import CATH
from IceCube_gen2_radio.EventTrace import EventTrace
from IceCube_gen2_radio.noise import *
# define ice parameters
n_index = 1
c = constants.c * units.m / units.s
n_firn = 1.3
n_air = 1.00029
n_ice = 1.74
prop = propagation.get_propagation_module('analytic')

#for the Summit station
ref_index_model = 'greenland_simple'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'GL1'


rays = prop(ice, attenuation_model,
            n_frequencies_integration=25,
            n_reflections=0)

def makefig(y, label='', xlabel='', ylabel='', savefig=''):
    plt.cla()
    fig, ax = plt.subplots()
    ax.plot(y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

# define detector layout
#Num_of_stations = 4
Num_of_stations = 1
array_of_st = np.empty(Num_of_stations, dtype=rno_g_st)

#array_of_st[0] = rno_g_st(21, np.array([-188 ,950, 0]) )
array_of_st[0] = rno_g_st(11, np.array([-1572.82, 729.37, -3.78]) )
# array_of_st[0] = rno_g_st(12, np.array([-1197.62, 2366.47, 0]) )
#array_of_st[3] = rno_g_st(22, np.array([14.789,	2162.74, 0]) )
#array_of_st[4] = rno_g_st(23, np.array([218.543,3375.16, 0])  )
#array_of_st[5] = rno_g_st(13, np.array([-993.895,	3578.9, 0]) )

# define CATH coordinates, antennas, antenna orientation, pulser trace - needs separate class
#cath_site = CATH(name='0', pulsing_mode ='IDL', coordinates=np.array([-197.75, 2295.1 , 5]))
#cath_site = CATH(name='1', pulsing_mode ='IDL', coordinates=np.array([-97.759100, 1915.1097 , 5]))
#cath_site = CATH(name='2', coordinates=np.array([-450.218, 1907.4 , 5]))
#cath_site = CATH(name='3', pulsing_mode ='IDL', coordinates=np.array([-238.953, 1150.35 , 5]))
#cath_site = CATH(name='4', coordinates=np.array([-693.31, 1658.41 , 5]))
#cath_site = CATH(name='6', pulsing_mode ='CW', coordinates=np.array([68.81, 532.7049999999999, 5]))
#cath_site = CATH(name='DISC', pulsing_mode ='IDL', coordinates=np.array([168.81, 432.705, 5]))

#cath_site = CATH(name='SCO proposal', pulsing_mode ='IDL', coordinates=np.array([198.81, 511.705, 5]))

cath_site = CATH(name='Helper String C calpulser', pulsing_mode ='rno_cal', coordinates=np.array([-1572.82, 763.87, 0]), depth=93.914)
#cath_site = CATH(name='cath 1', coordinates=np.array([40, 500, 5]))

freq_range = cath_site.get_pulse_fft_freq()
print(f'\n freq_range: length {len(freq_range)}, min {np.min(freq_range)}, max {np.max(freq_range)}')

freq_range_noise = np.arange(freq_range[0], freq_range[-1],
                             (freq_range[-1] - freq_range[0]) / 1025)
print(f'\n freq_range_noise = {freq_range}')

# define a single event - need event class, would contain traces for each channel and timing information
Event = np.empty(Num_of_stations,
                 dtype=EventTrace)  # for now, we only work with a single event, but each station has it's own Event object

##################################
# Pick CATH Pulsing Mode:
#pulsingMode = Vpol and Hpol # maybe will add CW mode in the future
pulsingMode = 'Vpol'
#################################
surf_ch_num = np.array([12,13,14,15,16,17,18,19,20])
# loop through every station
for st_id in range(0, Num_of_stations):
    ev_id = st_id
    Event[ev_id] = EventTrace(ev_id, array_of_st[st_id]._name)
    array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_surface_antenna)
    print('st_name:', array_of_st[st_id]._name, ' dist:', math.dist(array_of_st[st_id]._coordinates, cath_site._coordinates))
    #sampling_rate_for_IFFT = 2 * np.max(freq_range)
    sampling_rate_for_IFFT = 3.2 #GHz
    fiber_prop_time = 5 * 100
    #trigger_time = travel_time_inAir + cable_prop_time # take snapshot from the deep channels
    # loop through deep channels: **note maybe start with a single one?

    v_pols = np.array([0,1,2,3,5,6,7,9,10,22,23])
    h_pols = np.array([4,8,11,21])

    deep_channels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    #trigger_time =
    #deep_channels = np.concatenate((v_pols, h_pols), axis=0)
    # has to calculate trigger time ...
    travel_time = np.zeros(24)
    for count, ch_id in enumerate(deep_channels):
        # find travel distance: deep Tx and deep channel coordinate difference --
        if pulsingMode == 'Vpol':
            array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_vpol)
        if pulsingMode == 'Hpol':
            array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_hpol)

        initial_point = cath_site._coordinates_deep_antenna_vpol
        #final_point = array_of_st[st_id].getAntennaCoordinate()[ch_id]
        final_point = array_of_st[st_id].get_rno_st_antenna_coord_fromJson(array_of_st[st_id]._name)[ch_id]

        # run analytic ray tracing
        print('Tx,Rx:', initial_point, final_point, ch_id)
        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()
        n = rays.get_number_of_solutions()

        WF = [np.array([]) for i in range(n)]

        dt_DnR=0
        dt_channel = 0

        # loop through ray tracing solutions: *start with direct ray
        if n != 0:
            print('ch_id has solutions', ch_id, st_id)
            for iS in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(iS)
                solution_type = solution_types[solution_int]

                # save travel distance and time for each solution
                path_length = rays.get_path_length(iS)
                timeOfFlight = rays.get_travel_time(iS)
                bf_time_shift = (path_length/1000)*(-10)#ns

                if iS == 0:
                    travel_time[ch_id] = rays.get_travel_time(iS)
                    print('direct ray travel time', rays.get_travel_time(iS))
                    #dt_channel = travel_time - trigger_time - delayBW_SurfDeep
                    # if dt_channel <=0 :
                    #     print('Warning! signal reached the deep channel ' + str(ch_id) + ' before the trigger ')
                    #     print('Try set delayBW_SurfDeep closer to :', travel_time - trigger_time)
                if iS == 1:
                    dt_DnR = rays.get_travel_time(iS) - travel_time[ch_id]


                # find launching and receiving vectors for antenna radiation pattern
                launch_vector = rays.get_launch_vector(iS)
                receive_vector = rays.get_receive_vector(iS)

                zenith = hp.cartesian_to_spherical(*launch_vector)[0]
                azimuth = hp.cartesian_to_spherical(*launch_vector)[1]

                # calculate antenna radiation patter
                # apply cable attenuation on power spectrum
                # if cath_site._pulsing_mode != 'IDL' or 'CW':
                #     fft_idl = cath_site.get_pulse_fft() * cable_attenuation(freq_range, 200)
                #     print('taking into account cable attenuatio..')
                # else: fft_idl = cath_site.get_pulse_fft()
                fft_idl = cath_site.get_pulse_fft()
                if pulsingMode == 'Vpol':
                    antenna_orientation = cath_site.inIce_Tx_vpol_orientation
                    VEL = cath_site.antenna_inIce_Tx_vpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                          *antenna_orientation)
                if pulsingMode == 'Hpol':
                    antenna_orientation = cath_site.inIce_Tx_hpol_orientation
                    VEL = cath_site.antenna_inIce_Tx_hpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                          *antenna_orientation)
                #plt.plot(freq_range,abs(VEL['theta']), label='before HP filter')

                # VEL['theta']=apply_butterworth(VEL['theta'],freq_range, 0.195)
                # VEL['phi'] =apply_butterworth(VEL['phi'],freq_range, 0.195)


                # VEL['phi'] = VEL['theta'] * 0.1
                print('antenna response:', VEL['theta'])
                # save VEL['theta'] to a txt file
                np.savetxt(f'antenna_response_ch.txt', VEL['theta'])
                # if iS == 0:
                #     makefig(VEL['theta'], ylabel='antenna', label='theta', savefig=f'plots/emitter_VEL{ch_id}.png')
                eTheta = VEL['theta'] * (-1j) * fft_idl * freq_range * n_ice / (c)
                ePhi = VEL['phi'] * (-1j) * fft_idl * freq_range * n_ice / (c)
                if iS == 0:
                    makefig(eTheta, ylabel='post Tx antenna', label='eTheta', savefig=f'plots/eTheta_ch{ch_id}.png')

                eTheta = eTheta / path_length
                eTheta *= np.exp(-(-1j) * 2 * np.pi * freq_range * bf_time_shift) #birefringence
                ePhi = ePhi / path_length

                # propagate through the ice - apply ice attenuation and distance factors
                attenuation_ice = rays.get_attenuation(iS, freq_range, sampling_rate_for_IFFT)

                zenith = hp.cartesian_to_spherical(*receive_vector)[0]
                azimuth = hp.cartesian_to_spherical(*receive_vector)[1]

                if ch_id in v_pols:
                    RxAntenna_type = array_of_st[st_id].antenna_inIce_vpol
                    RxAntenna_orientation = array_of_st[st_id].getAntennaRotation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                else:
                    RxAntenna_type = array_of_st[st_id].antenna_inIce_hpol
                    RxAntenna_orientation = array_of_st[st_id].getAntennaRotation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                efield_antenna_factor = np.array([VEL['theta'], VEL['phi']])

                #Fold E-field with in ice Rx antennas
                power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
                power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0) * attenuation_ice
                power_spectrum_atRx = apply_butterworth(power_spectrum_atRx,freq_range, 0.100)
                
                if iS == 0:
                    makefig(power_spectrum_atRx, ylabel='post Rx antenna', label='power_spectrum_atRx', savefig=f'plots/postant-spec_ch{ch_id}.png')
                if iS == 0:
                    makefig(fft.freq2time(power_spectrum_atRx, sampling_rate_for_IFFT), ylabel='post Rx antenna volt', label='WF', savefig=f'plots/postant-time_ch{ch_id}.png')
                max_amplitude = np.max(fft.freq2time(power_spectrum_atRx, sampling_rate_for_IFFT))
                if max_amplitude > 1.1*1e-3 :
                    print('Channel is saturated!', ch_id, st_id)

                # Add thermal noise
                # arange new array of frequencies to generate 2048 samples of thermal noise
                # freq_range_noise = np.arange(freq_range[0], freq_range[-1],
                #                              (freq_range[-1] - freq_range[0])/1025 )
                thermal_noise_spectrum = generate_thermal_noise(freq_range_noise, depth = -final_point[2])

                #power_spectrum_atRx += thermal_noise_spectrum
                # apply amplifier
                amplifier_s11 = array_of_st[st_id].amp_response_iglu
                amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
                amp_response_noise = amplifier_s11['gain'](freq_range_noise) * amplifier_s11['phase'](freq_range_noise)
                # Add Amplifier noise
                ampl_noise =  get_noise_figure_IGLU_DRAP(freq_range_noise)
                #power_spectrum_atRx += ampl_noise
                # Fiber Link passes signal to the DAQ
                power_spectrum_atRx *= amp_response * fiber_link(freq_range, -final_point[2])
                thermal_noise_spectrum *= amp_response_noise * fiber_link(freq_range_noise, -final_point[2])
                ampl_noise *= amp_response_noise * fiber_link(freq_range_noise, -final_point[2])

                # fold signal with receiving antenna response

                WF[iS] = fft.freq2time(power_spectrum_atRx,
                                       sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
                # thermal_noise = fft.freq2time(thermal_noise_spectrum, sampling_rate_for_IFFT)
                makefig(WF[iS], ylabel='WF', label=f'Vpk-pk {(np.max(WF[iS])-np.min(WF[iS]))/2}', savefig=f'plots/WF_ch{ch_id}_sol{iS}.png')

                # WF[iS] += thermal_noise
                # if ch_id == 23:
                #     plt.plot(abs(VEL['theta']), label='before')
                #     plt.legend()
                #     plt.show()
            # Superimpose direct and reflected rays

            WF_DnR_superimposed = superimposeWF(WF, dt_DnR)


            # Shift traces in time with respect to trigger time
            WF_superimposed_shifted = shift_trace(WF_DnR_superimposed, 0, sampling_rate_for_IFFT) # 0 is channel delay need to add later

            WF_noise = fft.freq2time(thermal_noise_spectrum, 2 * np.max(freq_range_noise)) + \
                       fft.freq2time(ampl_noise, 2 * np.max(freq_range_noise))
            max_amplitude = np.max(WF_superimposed_shifted)
            rms_noise = np.std(WF_noise)
            #print('st_id:', array_of_st[st_id]._name,  'ch_id:', ch_id, np.round(max_amplitude/rms_noise,2 ) )


            Event[ev_id].set_trace(ch_id, WF_superimposed_shifted+WF_noise)
            #Event[ev_id].set_trace(ch_id, WF_DnR_superimposed + WF_noise)
        if n == 0:
            travel_time[ch_id] = +100000
    trig_channel_starting_time = np.min(travel_time[deep_channels])
    travel_time = travel_time - trig_channel_starting_time

    print('travel times:', travel_time[deep_channels])
    print('hit time,' , np.min(travel_time[deep_channels]))
    for count, ch_id in enumerate(deep_channels):
        print('channel shift time, ch id :', travel_time[ch_id], ch_id )
        trace_unshifted = Event[ev_id].get_trace(ch_id)
        trace_shifted = shift_trace(trace_unshifted, travel_time[ch_id], sampling_rate_for_IFFT)
        # plt.plot(trace_unshifted, label='before')
        # plt.legend()
        # plt.plot(trace_shifted, label='after')
        # plt.legend()
        # plt.show()
        Event[ev_id].set_trace(ch_id, trace_shifted)
        vpkpk = (np.max(trace_unshifted)-np.min(trace_unshifted))/2 #units V
        noise_wf = np.concatenate((trace_unshifted[:np.argmax(trace_unshifted)-200], trace_unshifted[np.argmax(trace_unshifted)+200:]))
        noise = np.sqrt(np.mean(noise_wf**2)) 
        snr = vpkpk/noise
        makefig(trace_unshifted, ylabel='V', label=f'ch{ch_id}\nSNR {(snr):.2f}\nVpk-pk {(vpkpk*1e3):.2f}mV\nnoise {(noise*1e3):.2f}mV', savefig=f'plots/final_ch{ch_id}.png')

# draw diagnostic plots - amplitude of signal in surface and deep channels - 2 plots, time delay between hit time in
#plot_surface_array(array_of_st, cath_site, Event, trigg_ch=0)

for st_id in range(0, Num_of_stations):
    ev_id = st_id
    #drawTraceSurfaceArrayRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)
    drawTraceDeepChannelsRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
                      Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length, cath_site._pulsing_mode, cath_site._name,cath_site._coordinates_deep_antenna_vpol[2])
    #drawFFTDeepChannelsRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
    #                      Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length)
    for ch in range(9):
        makefig(Event[ev_id].get_trace(ch) * 1e3, ylabel='mV', label=f'Vpk-pk {(np.max(Event[ev_id].get_trace(ch)) - np.min(Event[ev_id].get_trace(ch))) * 1e3 / 2:.2f}', savefig=f'plots/cath_{ch}.png')
        makefig(Event[ev_id].get_fft_trace(ch), ylabel='FFT [AU]', label='FFT', savefig=f'plots/cath_fft_{ch}.png')
        # plt.clf()
        # plt, ax = plt.subplots(2,1)
        # ax[0].plot(Event[ev_id].get_trace(ch) * 1e3, label=f'Vpk-pk {(np.max(Event[ev_id].get_trace(ch)) - np.min(Event[ev_id].get_trace(ch))) * 1e3 / 2:.2f} mV')
        # ax[0].set_xlabel('Time (ns)')
        # ax[0].set_ylabel('Voltage (mV)')
        # ax[0].legend()
        # ax[0].grid()
        # ax[1].plot(Event[ev_id].get_trace(ch), label=f'FFT')
        # ax[1].grid()
        # ax[1].set_ylabel('FFT [AU]')
        # plt.savefig(f'plots/cath_{ch}.png', bbox_inches='tight')
