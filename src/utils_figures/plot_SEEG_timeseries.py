'''
Make figures of SEEG timeseries for paper
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mne
import sys
import colorednoise as cn
from utils import get_ses_and_task, highpass_filter, compute_slp_sim, compute_onset, compute_offset, \
    plot_signal, plot_datafeatures, plot_2d_binary_SEEG, plot_SO_SP
from utils_simulate import  read_one_seeg_re_iis

sys.path.append('/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret, vep_prepare
# sys.path.append('/Users/dollomab/MyProjects/Epinov_trial/vep_run_Trial/fit/')
# import vep_prepare


def plot_SEEG_choi_timeseries(t, y, choi, ch_names, seeg_info=None, scaleplt=0.002, datafeature=None,
                              onsets=None, offsets=None, show_time_axis=True, t_range_step=5,
                              title='SEEG empirical recording'):
    ''' Plots SEEG timeseries for defined channels of interest
        t: time
        y: brain activity time series
        choi: channels of interest
        ch_names: list of all channels
    [optional]:
        seeg_info: dictionnary containing seizure onset and offset
        datafeature: envelope time series over the electrical brain signal
        onsets: estimated seizure onset per channel
        offsets: estimated seizure offset per channel
        title: plot title '''

    assert choi is not None

    fig = plt.figure(figsize=(40, 20))
    i = 0
    for ch in choi:
        if ch in ch_names:
            ind = ch_names.index(ch)
            plt.plot(t, scaleplt * (y[ind, :] - y[ind, 0]) + i, 'blue', lw=1)                   # plot SEEG signal
            if datafeature is not None:
                plt.plot(t, 0.2*(datafeature.T[ind]) + i, 'mediumorchid', lw=4)                          # plot datafeatures
            if onsets is not None:
                plt.scatter(onsets[ind], i, marker='o', color='blue', s=1200)
            if offsets is not None:
                plt.scatter(offsets[ind], i, marker='X', color='blue', s=2000)

            i += 1

    if seeg_info is not None:                                                                   # plot onset and offset
        vlines = [seeg_info['onset'], seeg_info['offset']]
        for x in vlines:
            plt.axvline(x, color='DeepPink', lw=3)

    if show_time_axis:
        t_range = np.arange(t[0], t[-1], t_range_step)                                          # show time points
        plt.xticks(t_range, np.linspace(0, t[-1] - t[0], t_range.shape[0], dtype='int'), fontsize=70)
    else:
        plt.xticks([])                                                                          # hide time points
    plt.ylim([-1, len(choi) + 0.5])
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(len(choi)), choi, fontsize=70)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.xlabel('Time [s]', fontsize=80)
    plt.ylabel('Electrodes', fontsize=80)
    plt.title(title, fontweight='bold', fontsize=80)
    plt.tight_layout()
    plt.show()

def plot_SEEG_timeseries(t, y, ch_names, seeg_info=None, scaleplt=0.002):
    '''Plots all SEEG timeseries'''
    fig = plt.figure(figsize=(40, 80))

    for ind, ich in enumerate(ch_names):
        plt.plot(t, scaleplt * (y[ind, :]) + ind, 'blue', lw=0.5);

    if seeg_info is not None:
        vlines = [seeg_info['onset'], seeg_info['offset']]
        for x in vlines:
            plt.axvline(x, color='DeepPink', lw=3)

    plt.xticks(fontsize=26)
    plt.ylim([-1, len(ch_names) + 0.5])
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=26)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.xlabel('Time', fontsize=50)
    plt.ylabel('Electrodes', fontsize=50)
    plt.title('SEEG recording', fontweight='bold', fontsize=50)
    plt.tight_layout()
    plt.show()


def main():
    # Select subject
    pid = 'id010_cmn' #'id008_dmc'
    pid_bids = f'sub-{pid[2:5]}'
    subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/'
    type_SEEG = 'stimulated'  # ['interictal', 'spontaneous', 'stimulated']

    if type_SEEG == 'stimulated':
        # bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohortStim/'
        # bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/ControlCohortStimAmplitude/'
        bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/ControlCohortStimLocation/'


    else:
        bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'

    sub_dir_retro = f'{subjects_dir}/{pid}'
    sub_dir_bids = f'{bids_path}/{pid_bids}'

    #%% Loading empirical SEEG data file
    if type_SEEG == 'spontaneous' or type_SEEG == 'stimulated':
        szr_name = f'{sub_dir_retro}/seeg/fif/CMN_criseStimTrainTB\'1-2P_140321B-BEX_0010.json'      # spontaneous data
        #DMC_criseStimB'2-3_161115B-BEX_0003.json
        seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(sub_dir_retro, szr_name) # load seizure data
        base_length = 8                                        # plot ts_on sec before and after
        start_idx = int((seeg_info['onset'] - base_length) * seeg_info['sfreq'])
        base_length = 10
        end_idx = int((seeg_info['offset'] + base_length) * seeg_info['sfreq'])
    elif type_SEEG == 'interictal':
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
    plot_SEEG_timeseries(t, y, ch_names, seeg_info)


    #%% Loading synthetic SEEG data file
    szr_index = 9 # TODO change
    clinical_hypothesis = False

    run = szr_index + 1
    ses, task = get_ses_and_task(type=type_SEEG)
    if clinical_hypothesis:
        acq = "clinicalhypothesis"
    else:
        acq = "VEPhypothesis"
    print('ses ' + str(ses) + ' ' + task + ' ' + acq + ' run', run)
    # sim_szr_name = f'{sub_dir_bids}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'
    sim_szr_name = f'{sub_dir_bids}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_group_4_run-0{run}_ieeg.vhdr'
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
    y_new = y_sim_AC + noise1*0.2 + noise2 * 0.1

    # Plot all simulated timeseries
    plot_SEEG_timeseries(t_sim, y_new, ch_names_sim, scaleplt=0.09, seeg_info=None)

    # Asserting the list is the same for both emp and sim, otherwise need to rewrite some of the following bits
    assert ch_names == ch_names_sim

    #%% Selecting a subset of channels of interest to plot
    # choi = ["B'3-4", "TB'5-6", "A'3-4", "OF'6-7", "OR'2-3", "OR1-2", "B1-2"]
    choi = ["TB'2-3", "TB'4-5", "A'1-2", "C'4-5", "TP'3-4", "TP'5-6", "C1-2"]
    # Computing their indexes in the global channel list
    choi_index = [] # TODO
    for ch in choi:
        if ch in ch_names:
            choi_index.append(ch_names.index(ch))

    # Plot empirical choi
    if type_SEEG == 'interictal':
        seeg_info = None
    plot_SEEG_choi_timeseries(t, y, choi, ch_names, seeg_info, scaleplt=0.0012, show_time_axis=False,
                              title='SEEG empirical time series')
    # Plot synthetic choi
    if type_SEEG == 'interictal':
        seeg_info_sim = None
    else:
        seeg_info_sim = {'onset': 5.5, 'offset': 50}
    plot_SEEG_choi_timeseries(t_sim, y_new, choi, ch_names_sim, seeg_info_sim,
                              scaleplt=0.15, show_time_axis=False, title='SEEG simulated time series')



    #%% Compute data features
    # Empirical
    hpf = 30
    lpf = 0.06
    ts_on = base_length
    ts_off = base_length
    slp = vep_prepare.compute_slp(seeg_info, bip, hpf=hpf, lpf=lpf, ts_on=ts_on, ts_off=ts_off)
    removebaseline = True
    if removebaseline:
        ts_cut = ts_on / 4
        baseline_N = int((ts_on - ts_cut) * seeg_info['sfreq'])
        slp = slp - np.mean(slp[:baseline_N, :], axis=0)
    plot_datafeatures(slp)
    # Compute seizure onset and offset for each channel
    start = int(ts_on * seeg_info['sfreq'])
    end = int((ts_on + seeg_info['offset'] - seeg_info['onset']) * seeg_info['sfreq'])
    onsets = compute_onset(slp, start, end, thresh=0.01)
    offsets = compute_offset(slp, start, end, thresh=0.0001)
    plot_SEEG_choi_timeseries(t, y, choi, ch_names, seeg_info, scaleplt=0.001, datafeature=slp,
                              onsets=t[onsets], offsets=t[offsets], show_time_axis=False, title='SEEG empirical time series')

    # Simulated
    hpf_sim = 400
    lpf_sim = 2
    base_length_sim = 100 # TODO change
    slp_sim = compute_slp_sim(y_sim_AC.T, hpf=hpf_sim, lpf=lpf_sim, sfreq=1000)
    removebaseline = True
    if removebaseline:
        baseline_N = 100 # TODO change
        # int((ts_on - ts_cut) * seeg_info['sfreq'])
        slp_sim = slp_sim - np.mean(slp_sim[:baseline_N, :], axis=0)
    plot_datafeatures(slp_sim)
    # Compute seizure onset and offset for each channel
    start_sim = int(0.4 * sfreq_sim)
    end_sim = int(2.9 * sfreq_sim) # slp_sim.shape[0] - base_length_sim
    onsets_sim = compute_onset(slp_sim, start_sim, end_sim, thresh=0.2)  # 0.1 TODO thresh changed here is that ok?
    offsets_sim = compute_offset(slp_sim, start_sim, end_sim, thresh=0.1)
    plot_SEEG_choi_timeseries(t_sim, y_new, choi, ch_names_sim, seeg_info={'onset': 0.4, 'offset':2.9}, scaleplt=0.08,
                              datafeature=slp_sim, onsets=t_sim[onsets_sim], offsets=t_sim[offsets_sim],
                              show_time_axis=False, title='SEEG simulated time series')


    #%%  Binarize SEEG timeseries into a 2D binary image, where 0: no seizure, 1: seizure
    # we only take into account the window from seizure onset until seizure offset (so we remove the base length)

    # Empirical
    # compute peak enveloppe values for each channel
    base_length_sfreq = int(base_length * seeg_info['sfreq'])
    peak_vals = np.max(slp[base_length_sfreq:-base_length_sfreq, :], axis=0)

    plot_binary = True
    start = int(base_length * seeg_info['sfreq'])
    end = slp.shape[0] - int(base_length * seeg_info['sfreq'])
    binarized2Dslp = np.zeros(shape=slp[start:end, :].shape, dtype=int)
    for ch_idx in range(len(ch_names)):
        if peak_vals[ch_idx] > 2:    # if True, seizure channel
            binarized2Dslp[onsets[ch_idx]-start:offsets[ch_idx]-start, ch_idx] = 1 # TODO run analysis again on this new method
    if plot_binary:
        plot_2d_binary_SEEG(binarized2Dslp[:, choi_index], choi, scaleplt=0.01, fig_size=(40,20), marker_size=5000)

    # Simulated
    # compute peak enveloppe values for each channel
    peak_vals_sim = np.max(slp_sim[base_length_sim:-base_length_sim, :], axis=0)
    peak_thresh_ez = 4

    t_min_seizure_start_sim = min([t_sim[onsets_sim[i]] for i in np.where(peak_vals_sim > peak_thresh_ez)[0]])
    start_sim = int(t_min_seizure_start_sim * sfreq_sim)
    t_max_seizure_end_sim = max([t_sim[offsets_sim[i]] for i in np.where(peak_vals_sim > peak_thresh_ez)[0]])
    end_sim = int(t_max_seizure_end_sim * sfreq_sim)
    binarized2Dslp_sim = np.zeros(shape=slp_sim[start_sim:end_sim, :].shape, dtype=int)
    for ch_idx in range(len(ch_names_sim)):
        if peak_vals_sim[ch_idx] > peak_thresh_ez:   # if True, seizure channel
            binarized2Dslp_sim[onsets_sim[ch_idx] - start_sim:offsets_sim[ch_idx] - start_sim, ch_idx] = 1  # TODO run analysis again on this new method
    if plot_binary:
        plot_2d_binary_SEEG(binarized2Dslp_sim[:, choi_index], choi, scaleplt=0.1, fig_size=(40,20), marker_size=5000)


    # Plotting choi's datafeatures
    data = slp_sim[:, choi_index]
    fig = plt.figure(tight_layout=True)
    plt.plot(data, linewidth=2.5, color='mediumorchid')
    plt.hlines(xmin=0, xmax=data.shape[0], y=5, colors='black', linewidth=3)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Plotting choi's SO and SP
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    plt.subplot(gs[0])
    plt.ylabel('Electrodes', fontsize=36)
    EZ_channels = choi[:3]
    PZ_channels = choi[3:]
    plot_SO_SP(choi, EZ_channels, PZ_channels, x_label='Empirical')
    plt.subplot(gs[1])
    EZ_channels_sim = choi[:3]
    PZ_channels_sim = choi[3:]
    PZ_channels_sim.remove("OR'2-3")
    plot_SO_SP(choi, EZ_channels_sim, PZ_channels_sim, x_label='Simulated')
    plt.show()

# TODO maybe do this for another patient if needed : id015 seems quite good
