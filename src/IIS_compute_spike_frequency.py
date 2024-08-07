'''
Computes spike frequency per channel for both empirical and simulated SEEG interictal spike data
'''
import sys
import mne
import os
import pandas as pd
from virtual_epileptic_cohort.src.utils import *
from virtual_epileptic_cohort.src.utils_simulate import read_one_seeg_re_iis

sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret
roi = vep_prepare_ret.read_vep_mrtrix_lut()

sys.path.insert(2, '/Users/dollomab/MyProjects/Epinov_trial/interictal_patients/src/')
import detect_spikes

from scipy.signal import butter, sosfiltfilt
def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    sets up a bandpass butterworth filter
    figures out the butter filter order and high and lowpass coefficients
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def plot_spike_rate_emp_sim(spike_count_normalized, spike_count_normalized_sim, ch_names):
    ''' Plots computed spike rate for empirical and simulated SEEG timeseries '''
    nr_channels = len(ch_names)
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(nr_channels), spike_count_normalized, color='blue', alpha=0.4, label='Empirical spike rate')
    plt.bar(np.arange(nr_channels), spike_count_normalized_sim, color='violet', alpha=0.6, label='Simulated spike rate')
    plt.xticks(np.arange(nr_channels), ch_names, rotation=90, fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_spike_rate_correlation_emp_sim(spike_count_normalized, spike_count_normalized_sim):
    ''' Plots spike rate correlation between empirical and simulated SEEG timeseries
        returns computed correlation '''
    plt.figure()
    corr_spike_rate = calc_correlation(spike_count_normalized, spike_count_normalized_sim)
    plt.plot(spike_count_normalized, spike_count_normalized_sim, '.', label=f'correlation={round(corr_spike_rate, 2)}')
    plt.xlabel('Spike rate empirical')
    plt.ylabel('Spike rate simulated')
    plt.tight_layout()
    plt.legend()
    plt.show()

    return corr_spike_rate

def compute_mse_spike_rate_emp_sim(spike_count_normalized, spike_count_normalized_sim):
    A = spike_count_normalized
    B = spike_count_normalized_sim
    mse = ((A - B) ** 2).mean(axis=0)
    rmse = mse ** (0.5)
    return mse, rmse

def group_spike_rate(channel_names_grouped_dict, ch_names, spike_rate_per_channel):
    ''' Groups SEEG channels according to gain prior matrix
        returns average spike rate for each group of channels '''
    spike_rate_grouped_channels = []
    for ch_group in channel_names_grouped_dict.keys():
        channels = channel_names_grouped_dict.get(ch_group)
        sum_spike_rate = 0
        for ch in channels:
            idx = ch_names.index(ch)
            sum_spike_rate += spike_rate_per_channel[idx]
        spike_rate_grouped_channels.append(sum_spike_rate/len(channels))
    # return normalized spike rate
    return spike_rate_grouped_channels/np.sum(spike_rate_grouped_channels)

# ------------------------- Steps for EMPIRICAL interictal data ---------------------------------
# Step 1 read SEEG interictal spike data
retro_patients_path = '/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients'
patients_list = np.loadtxt(f'{retro_patients_path}/sublist.txt', dtype=str)
# stopped at id005
for i in range(15,30):
    i = 25
    pid = patients_list[i]
    print(pid)
    szr_type = 'interictal'
    szr_index = 0
    pid_bids = f'sub-{pid[2:5]}'
    subj_proc_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'

    save_path = f'{subj_proc_dir}/comparison_emp_sim_IIS'                             # place where to save figures
    os.makedirs(save_path, exist_ok=True)

    szr_names = load_seizure_name(pid, type=szr_type)                                 # load seizure name
    szr_name = f"{subj_proc_dir}/seeg/fif/{szr_names[szr_index].strip(' ')}.json"
    seeg_info, bip, gain, gain_prior = read_one_seeg_re_iis(subj_proc_dir, szr_name)  # read file content

    onset = 0                           # plot ts_on sec before and after
    offset = 900                        # taking 15 minutes here... is that ok ????
    start_idx = int(onset * seeg_info['sfreq'])
    end_idx = int(offset * seeg_info['sfreq'])
    scaleplt = 0.002
    y = bip.get_data()[:, start_idx:end_idx]
    t = bip.times[start_idx:end_idx]
    ch_names = bip.ch_names             # load electrode positions

    plot = False                        # plot empirical SEEG timeseries
    if plot:
        fig = plt.figure(figsize=(40, 80))
        nch_source = []
        for ind_ch, ichan in enumerate(ch_names):
            isource = roi[np.argmax(gain[ind_ch])]
            nch_source.append(f'{isource}:{ichan}')
        for ind, ich in enumerate(ch_names):
            plt.plot(t, scaleplt * (y[ind, :]) + ind, 'blue', lw=0.5)
        plt.xticks(fontsize=26)
        plt.ylim([-1, len(ch_names) + 0.5])
        plt.xlim([t[0], t[-1]])
        plt.yticks(range(len(ch_names)), [nch_source[ch_index] for ch_index in range(len(ch_names))], fontsize=26)
        plt.gcf().subplots_adjust(left=0.2, top=0.97)
        plt.xlabel('Time', fontsize=50)
        plt.ylabel('Electrodes', fontsize=50)
        plt.title('IIS recording', fontweight='bold', fontsize=50)
        plt.tight_layout()
        plt.show()

    # Step 2 band pass filter the data
    fs = seeg_info['sfreq']                     # Sample rate and desired cutoff frequencies in Hz
    lowcut = 1                                  # Hz
    highcut = 70                                # Hz
    sos = butter_bandpass(lowcut, highcut, fs)  # bandpass butterworth filter
    yfilt = sosfiltfilt(sos, y)                 # Use filtfilt to apply a noncasual filter

    # Step 3 spike detection
    # Considering a spike a signal which crosses a defined threshold AND that has a minimum amplitude of 25 microV
    nr_channels = y.shape[0]                                               # number of SEEG channels
    spike_count_all_channels = np.zeros(shape=(nr_channels))               # contains nr of spikes per each SEEG channel
    for ch_index in range(nr_channels):
        threshold_Q = np.median(np.absolute(yfilt[ch_index, :])/0.6745) * 5  # taken from Quian Quiroga et al. 2004
        threshold = max(25, threshold_Q)
        opti_spikes_idx, better_spikes_idx = detect_spikes.get_spikes_idx_realdata(yfilt[ch_index, :], threshold=threshold)
        spike_count_all_channels[ch_index] = len(opti_spikes_idx)

    # Step 4 calculate normalized spike rate per channel
    spike_count_normalized = spike_count_all_channels/np.sum(spike_count_all_channels)  # normalize the spike count/channel

    # Plotting results [optional]
    plot = False
    save_fig=True
    if plot or save_fig:
        plot_signal_ez_pz(t, y, spike_count_normalized, ch_names, ez_channels=ch_names,
                      pz_channels=[], scaleplt=scaleplt,
                      save_path=f'{save_path}/{pid_bids}_{szr_type}_timeseries_spikecount_emp.png')
    # Plotting some of the spikes [optional]
    if plot:
        ch_index = ch_names.index("OF'1-2")
        threshold = np.median(np.absolute(yfilt[ch_index, :]) / 0.6745) * 5
        opti_spikes_idx, better_spikes_idx = detect_spikes.get_spikes_idx(yfilt[ch_index, :], threshold=threshold)

        signalData = yfilt[ch_index, :]
        tData = t[:]
        spikes_idx = np.array(opti_spikes_idx, dtype=int)
        fig = plt.figure(figsize=(13,5))
        plt.plot(tData, signalData)
        plt.title("Epileptors time series and spike detection")
        plt.hlines(np.mean(signalData), t[0], t[-1], 'r')
        # plt.hlines(np.std(signalData)*3, 0, t[-1], 'orange', label='3 x standard deviation')
        # plt.hlines(-np.std(signalData)*3, 0,  t[-1], 'orange')
        plt.hlines(threshold, t[0], t[-1], 'green', label = 'Threshold')
        plt.hlines(-threshold, t[0], t[-1], 'green')
        plt.xticks(fontsize=12)
        # plt.ylim([-1,len(ez)+0.5])
        # plt.xlim([t[0],t[-1]])
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel(f'{ch_names[ch_index]}', fontsize=18)
        for el in t[spikes_idx]:
            plt.plot(el, 1, 'ro')
        plt.tight_layout()
        plt.legend()
        plt.show()

    # ------------------------ Redo all steps for SIMULATED interictal data ---------------------------------

    # clinical_hypothesis = False     #TODO change
    for clinical_hypothesis in [True, False]:

        database = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'
        sim_patient_data = f'{database}/{pid_bids}'
        run = szr_index + 1
        ses, task = get_ses_and_task(type='interictal')
        if clinical_hypothesis:
            acq = "clinicalhypothesis"
        else:
            acq = "VEPhypothesis"
        print('ses ' + str(ses) + ' ' + task + ' ' + acq + ' run', run)
        seizure_name = f'{sim_patient_data}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'
        raw = mne.io.read_raw_brainvision(seizure_name, preload=True)
        y_sim = raw._data
        y_sim_AC = highpass_filter(y_sim, 256, filter_order = 101)
        t_sim = raw.times
        ch_names_sim = raw.ch_names

        plot=False
        if plot:
            plot_signal(t_sim, y_sim_AC, ch_names_sim,  scaleplt=0.1)   # plotting simulated interictal timeseries

        # Spike detection
        nr_channels_sim = len(ch_names_sim)
        spike_count_all_channels_sim = np.zeros(shape=(nr_channels_sim))                 # contains nr of spikes per each SEEG channel
        for ch_index in range(nr_channels_sim):
            # print(ch_index)
            threshold = np.median(np.absolute(y_sim_AC[ch_index, :])/0.6745) * 5 # taken from Quian Quiroga et al. 2004 # TODO change
            opti_spikes_idx, better_spikes_idx = detect_spikes.get_spikes_idx(y_sim_AC[ch_index, :], threshold=threshold)
            spike_count_all_channels_sim[ch_index] = len(opti_spikes_idx)

        # normalize the spike count/channel
        spike_count_normalized_sim = spike_count_all_channels_sim/np.sum(spike_count_all_channels_sim)

        # Plotting results [optional]
        plot = False
        save_fig = True
        if plot or save_fig:
            plot_signal_ez_pz(t_sim, y_sim_AC, spike_count_normalized_sim, ch_names_sim, ez_channels=ch_names_sim,
                          pz_channels=[], scaleplt=0.1,
                          save_path=f'{save_path}/{pid_bids}_{szr_type}_{acq}_timeseries_spikecount_sim.png') # plotting results

        # Plot some of the timeseries and detected spikes [optional]
        plot = False
        if plot:
            ch_index = ch_names_sim.index("A3-4")
            threshold = np.median(np.absolute(y_sim_AC[ch_index, :]) / 0.6745) * 5
            opti_spikes_idx, better_spikes_idx = detect_spikes.get_spikes_idx(y_sim_AC[ch_index, :], threshold=threshold)

            signalData = y_sim_AC[ch_index, :]
            spikes_idx = np.array(opti_spikes_idx, dtype=int)
            fig = plt.figure(figsize=(13,5))
            plt.plot(t_sim, signalData)
            plt.title("Epileptors time series and spike detection")
            plt.hlines(np.mean(signalData), t_sim[0], t_sim[-1], 'r')
            # plt.hlines(np.std(signalData)*3, t_sim[0], t_sim[-1], 'orange', label='3 x standard deviation')
            # plt.hlines(-np.std(signalData)*3, t_sim[0],  t_sim[-1], 'orange')
            plt.hlines(threshold, t_sim[0], t_sim[-1], 'green', label = 'Threshold')
            plt.hlines(-threshold, t_sim[0], t_sim[-1], 'green')
            plt.xticks(fontsize=12)
            # plt.ylim([-1,len(ez)+0.5])
            # plt.xlim([t[0],t[-1]])
            plt.xlabel('Time [s]', fontsize=18)
            plt.ylabel(f'{ch_names_sim[ch_index]}', fontsize=18)
            for el in t_sim[spikes_idx]:
                plt.plot(el, 1, 'ro')
            plt.tight_layout()
            plt.legend()
            plt.show()

        # Step 5 comparison simulated vs recorded
        plot_spike_rate_emp_sim(spike_count_normalized, spike_count_normalized_sim, ch_names)
        correlation = plot_spike_rate_correlation_emp_sim(spike_count_normalized, spike_count_normalized_sim)
        mse, rmse = compute_mse_spike_rate_emp_sim(spike_count_normalized, spike_count_normalized_sim)
        print(f'Correlation={correlation}, MSE={mse}')

        assert len(ch_names_sim) == len(ch_names)
        assert ch_names == ch_names_sim
        channel_names_grouped_dict = group_channels(ch_names, gain_prior)   # grouping channels according to gain prior
        channel_names_grouped = list(channel_names_grouped_dict.keys())

        spike_count_grouped = group_spike_rate(channel_names_grouped_dict, ch_names, spike_count_all_channels)
        spike_count_grouped_sim = group_spike_rate(channel_names_grouped_dict, ch_names_sim, spike_count_all_channels_sim)

        # plot grouped results
        plot_spike_rate_emp_sim(spike_count_grouped, spike_count_grouped_sim, channel_names_grouped)
        correlation_grouped = plot_spike_rate_correlation_emp_sim(spike_count_grouped, spike_count_grouped_sim)
        mse_grouped, rmse_grouped = compute_mse_spike_rate_emp_sim(spike_count_grouped, spike_count_grouped_sim)
        print(f'Correlation grouped={correlation_grouped}, MSE grouped={mse_grouped}')


        # ------------------------ SAVE spike rate for EMPIRICAL and SIMULATED timeseries ---------------------------------
        # df = pd.DataFrame(columns=['subject_id', 'emp_interictal_fname', 'sim_interictal_fname',
        #                            'emp_spike_rate_per_channel', 'sim_spike_rate_per_channel', 'emp_ch_names',
        #                            'sim_ch_names', 'correlation', 'mse', 'grouped_correlation', 'grouped_mse'])
        # filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
        # df.to_csv(filepath, index=False)

        add_row = False          # add row at the end of the dataframe
        if add_row:
            filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
            df = pd.read_csv(filepath)

            new_row = {'subject_id': pid_bids, 'emp_interictal_fname': szr_names[szr_index].strip(" "),
                       'sim_interictal_fname': f'{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg',
                       'emp_spike_rate_per_channel': spike_count_normalized,
                       'sim_spike_rate_per_channel': spike_count_normalized_sim,
                       'emp_ch_names': ch_names, 'sim_ch_names': ch_names_sim,
                       'correlation': correlation, 'mse': mse,
                       'grouped_correlation': correlation_grouped, 'grouped_mse': mse_grouped}
            df2 = df.append(new_row, ignore_index=True)

            # save dataframe
            df2.to_csv(filepath, index=False)

        change_row = True         # change values in certain row
        row_id = 50
        if change_row:
            filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
            df = pd.read_csv(filepath)
            print(df.loc[row_id])

            df.loc[row_id, ['subject_id', 'emp_interictal_fname', 'sim_interictal_fname']] = [pid_bids, szr_names[szr_index].strip(" "),
                       f'{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg']

            df.loc[row_id, ['emp_spike_rate_per_channel', 'sim_spike_rate_per_channel']] = [str(spike_count_normalized), str(spike_count_normalized_sim)]

            df.loc[row_id, ['emp_ch_names', 'sim_ch_names']] = [str(ch_names), str(ch_names_sim)]
            df.loc[row_id, ['correlation', 'mse', 'grouped_correlation', 'grouped_mse']] = [correlation, mse, correlation_grouped,  mse_grouped]

            df.to_csv(filepath, index=False)





