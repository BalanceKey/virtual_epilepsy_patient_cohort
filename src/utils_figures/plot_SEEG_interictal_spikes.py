'''
Make figures of SEEG interictal spike timeseries for paper
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mne
import sys
import colorednoise as cn
from scipy.signal import sosfiltfilt, butter
from src.utils_simulate import  read_one_seeg_re_iis
from src.utils import get_ses_and_task, highpass_filter
from src.utils_figures.plot_SEEG_timeseries import plot_SEEG_choi_timeseries, plot_SEEG_timeseries


def main():

    # Select subject
    pid = 'id008_dmc'# 'id010_cmn'
    pid_bids = f'sub-{pid[2:5]}'
    subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/'
    bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'

    sub_dir_retro = f'{subjects_dir}/{pid}'
    sub_dir_bids = f'{bids_path}/{pid_bids}'

    type_SEEG = 'interictal' # ['interictal', 'spontaneous', 'stimulated']

    #%% Loading empirical SEEG data file
    szr_name = f'{sub_dir_retro}/seeg/fif/DMC_INTERC_161110B-DEX_0003.json'                      # interictal data
    seeg_info, bip, gain, gain_prior = read_one_seeg_re_iis(sub_dir_retro, szr_name)             # load seizure data
    onset = 0                                               # plot ts_on sec before and after
    offset = 200#900                                            # taking 15 minutes here
    start_idx = int(onset * seeg_info['sfreq'])
    end_idx = int(offset * seeg_info['sfreq'])

    y = bip.get_data()[:, start_idx:end_idx]
    t = bip.times[start_idx:end_idx]

    # load electrode positions
    ch_names = bip.ch_names

    # Plot all empirical time series
    plot_SEEG_timeseries(t, y, ch_names, seeg_info=None, scaleplt=0.004)


    #%% Loading synthetic SEEG data file
    clinical_hypothesis = False

    run = 1
    ses, task = get_ses_and_task(type=type_SEEG)
    if clinical_hypothesis:
        acq = "clinicalhypothesis"
    else:
        acq = "VEPhypothesis"
    print('ses ' + str(ses) + ' ' + task + ' ' + acq + ' run', run)
    sim_szr_name = f'{sub_dir_bids}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'
    raw = mne.io.read_raw_brainvision(sim_szr_name, preload=True)
    y_sim = raw._data
    y_sim_AC = highpass_filter(y_sim, 256, filter_order = 101)
    t_sim = raw.times
    ch_names_sim = raw.ch_names
    sfreq_sim = raw.info['sfreq']

    # Adding noise to y_sim
    beta = 1  # the exponent
    noise1 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
    beta = 2  # the exponent
    noise2 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
    beta = 3  # the exponent
    noise3 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
    # y_new = y_filt + noise + noise2
    y_new = y_sim_AC + noise1*4 + noise2 * 0.5

    # Plot all simulated timeseries
    plot_SEEG_timeseries(t_sim, y_sim_AC, ch_names_sim, scaleplt=0.08, seeg_info=None)


    # Asserting the list is the same for both emp and sim, otherwise need to rewrite some of the following bits
    assert ch_names == ch_names_sim

    #%% Selecting a subset of channels of interest to plot
    choi = ["B'3-4", "TB'5-6", "A'3-4", "OF'6-7", "OR'2-3", "OR1-2", "B1-2"]
    # Computing their indexes in the global channel list
    choi_index = [] # TODO
    for ch in choi:
        if ch in ch_names:
            choi_index.append(ch_names.index(ch))

    # Plot empirical choi
    if type_SEEG == 'interictal':
        seeg_info = None
    start = 0
    end = 120000#110000
    plot_SEEG_choi_timeseries(t[start:end], y[:, start:end], choi, ch_names, seeg_info, scaleplt=0.004, show_time_axis=True,
                              title='SEEG empirical time series')
    # Plot synthetic choi
    if type_SEEG == 'interictal':
        seeg_info = None
    else:
        seeg_info = {'onset': 0.4, 'offset': 2.9}

    start_sim = 0
    end_sim = 4000
    plot_SEEG_choi_timeseries(t_sim[start_sim:end_sim], y_new[:, start_sim:end_sim], choi, ch_names_sim, seeg_info,
                              scaleplt=0.02, show_time_axis=False, title='SEEG simulated time series')


    #%% Zoom-in into a spike and plot
    channel = "B'3-4"
    channel_index = ch_names.index(channel)
    channel_timeseries_emp = y[channel_index, :]
    channel_timeseries_sim = y_new[channel_index, :]

    # Band pass filter the data
    fs = seeg_info['sfreq']                     # Sample rate and desired cutoff frequencies in Hz
    lowcut = 1                                  # Hz
    highcut = 40                                # Hz
    # bandpass butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    order = 5
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    yfilt = sosfiltfilt(sos, channel_timeseries_emp)                 # Use filtfilt to apply a noncasual filter


    start = 4500
    end = 7200
    plt.figure(figsize=(3,5), tight_layout=True)
    plt.plot(channel_timeseries_emp[start:end])
    plt.show()

    plt.figure(figsize=(3,5), tight_layout=True)
    plt.plot(yfilt[start:end])
    plt.show()

    start_sim = 131
    end_sim = 193
    plt.figure(figsize=(3,5), tight_layout=True)
    plt.plot(channel_timeseries_sim[start_sim:end_sim])
    plt.show()

