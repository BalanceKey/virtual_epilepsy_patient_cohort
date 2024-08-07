import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.transforms import Affine2D
from matplotlib.ticker import StrMethodFormatter
import sys
sys.path.append('/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare

from sklearn.decomposition import PCA

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_PCA_components(y, ch_names, plot=False):
    pca_all = PCA(n_components=len(ch_names))
    pca_all.fit(y)
    # comp = pca_all.components_
    if plot:
        npc = np.argmin(np.gradient(pca_all.explained_variance_ratio_)) + 1
        plt.plot(pca_all.explained_variance_ratio_, '.')
        plt.axvline(npc, color='k', alpha=0.5)
        plt.show()
    return pca_all

def feature_vector_dim(pca_all, max_variance=0.8):
    ''' returns the number of principal components required to explain  max_variance (e.g. 80)% of the variance
        aka the dimensionality of the feature vector '''
    explained_variance_sum = 0
    for i, eigenval in enumerate(pca_all.explained_variance_ratio_):
        explained_variance_sum += eigenval
        if explained_variance_sum >= max_variance:
            return i

def get_ses_and_task(type='spontaneous'):
    ''' From the type of seizure gives the BIDS ses nr and task name '''
    if type == 'spontaneous':
        ses = 1
        task = "simulatedseizure"
    elif type == 'stimulated':
        ses = 2
        task = "simulatedstimulation"
    elif type == 'interictal':
        ses = 3
        task = 'simulatedinterictalspikes'
    return ses, task

def rename_brainvision_file(vhdr_file_path, vhdr_file_renamed_path):
    ''' Renames a brainvision file to the new name. Absolute paths are given in the arguments. '''
    from mne_bids.copyfiles import copyfile_brainvision
    copyfile_brainvision(vhdr_file_path, vhdr_file_renamed_path, verbose=True)

def load_seizure_list(subj_dir):
    subj_seizures = f'{subj_dir}/seeg/fif/'
    status = subprocess.run(f'ls {subj_seizures}/*.json', shell=True, capture_output=True)
    out = status.stdout.decode('utf-8').strip().split('\n')
    seizures = [s.split('/')[-1].split('.')[0] for s in out]
    return seizures

def load_seizure_name(pid, type='spontaneous'):
    ''' Loads a seizure name(s) for a patient and a give type : spontaneous, stimulated, interictal '''
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
    df = pd.read_csv(filepath)
    patient_index = df.index[df['subject_id'] == pid[:5]][0]
    if type == 'spontaneous':
        seizure_names = df.at[patient_index, 'sim_spontaneous']
    elif type == 'stimulated':
        seizure_names = df.at[patient_index, 'sim_stimulated']
    elif type == 'interictal':
        seizure_names = df.at[patient_index, 'sim_interictal']
    else:
        print('Error in seizure selection check type !')
        seizure_names = None
    return seizure_names.strip("[] ").split(',')

def highpass_filter(y, sr, filter_order = 501):
    """In this case, the filter_stop_freq is that frequency below which the filter MUST act like a s
    top filter and filter_pass_freq is that frequency above which the filter MUST act like a pass filter.
       The frequencies between filter_stop_freq and filter_pass_freq are the transition region or band."""
    filter_stop_freq = 3  # Hz
    filter_pass_freq = 3  # Hz
    filter_order = filter_order
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    # Apply high-pass filter
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

def calc_correlation(actual, predic):
    '''Computes pearson correlation between two vectors (if I'm not mistaken)'''
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator

def plot_signal(t, y, ch_names, seeg_info=None, scaleplt=0.001,
                datafeature=None, ez_channels_slp=None, onsets=None, offsets=None, save_path=None):
    '''
    Simple plot of the empirical/simulated SEEG
    '''
    fig = plt.figure(figsize=(40, 80))
    for ind, ich in enumerate(ch_names):
        plt.plot(t, scaleplt * (y[ind, :]-y[ind, 0]) + ind, 'blue', lw=0.5)
        if datafeature is not None:
            if ez_channels_slp is not None:
                plt.plot(t, 0.5 * (datafeature.T[ind] - datafeature.T[ind, 0]) + ind, color=(1.0 * ez_channels_slp[ind], 0.0, 0.0), lw=1.5)
            else:
                plt.plot(t, 0.5*(datafeature.T[ind] - datafeature.T[ind,0]) + ind, 'red', lw=1.5)
        if onsets is not None:
            plt.scatter(onsets[ind], ind, marker='o', color='green', s=250)
        if offsets is not None:
            plt.scatter(offsets[ind], ind, marker='o', color='red', s=250)

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
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def plot_signal_ez_pz(t, y, peak_vals, ch_names, ez_channels, pz_channels, seeg_info=None, scaleplt=0.001, save_path=None):
    '''
    Plots SEEG signal alonside signal power and EZ PZ estimated electrodes (different colors in the histogram)
    '''
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    fig = plt.figure(figsize=(40, 80))
    plt.subplot(gs[0])

    for ind, ich in enumerate(ch_names):
        plt.plot(t, scaleplt * (y[ind, :]-y[ind, 0]) + ind, 'blue', lw=0.5)
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
    plt.subplot(gs[1])
    colors = np.empty(shape=peak_vals.shape, dtype='str')
    for i, ch in enumerate(ch_names):
        if ch in ez_channels:
            colors[i] = 'b'
        elif ch in pz_channels:
            colors[i] = 'c'
        else:
            colors[i] = 'w'
    plt.barh(np.r_[1:peak_vals.shape[0] + 1], peak_vals, color=colors, edgecolor='k', alpha=0.4, log=False)
    plt.ylabel('Electrodes', fontsize=50)
    plt.xlabel('EnvAmplitude', fontsize=50)
    plt.xticks(fontsize=26)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.yticks(np.r_[1:len(ch_names) + 1], ch_names, fontsize=26)
    plt.ylim([0, len(ch_names) + 1])
    plt.title('SEEG enveloppe amplitude', fontweight='bold', fontsize=50)
    plt.tight_layout()
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def compute_slp_sim(seeg, hpf=10.0, lpf=1.0, sfreq=1000, remove_outliers=False):
    '''
    Computes the data feature for the SEEG (empirical/simulated)
    '''
    y = seeg
    slp = y
    # Remove outliers i.e data > 2*sd
    if remove_outliers:
        for i in range(slp.shape[1]):
            ts = slp[:, i]
            ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()

    # High pass filter the data
    if hpf is not None:
        slp = vep_prepare.bfilt(slp, sfreq, hpf, 'highpass', axis=0)

    # Compute seeg log power
    slp = vep_prepare.seeg_log_power(slp, 100)

    # Remove outliers i.e data > 2*sd
    if remove_outliers:
        for i in range(slp.shape[1]):
            ts = slp[:, i]
            ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()

    # Low pass filter the data to smooth
    if lpf is not None:
        slp = vep_prepare.bfilt(slp, sfreq, lpf, 'lowpass', axis=0)
    return slp

def compute_onset(slp, start, end, thresh=0.1):
    ''' Computes and approximated seizure onset per channel by first computing
        the peak of the envelope and then finding the last point in the envelope
        prior to the peak that has less than 10% of the peak amplitude '''
    # start = int(ts_on * seeg_info['sfreq'])
    # end = slp.shape[0]  # int((ts_on + 138) * seeg_info['sfreq'])
    onsets = np.empty(shape=slp.shape[1], dtype=int)
    for i in range(slp.shape[1]):
        peak_index = np.argmax(slp[start:end, i])
        peak_value = np.max(slp[start:end, i])
        onset = np.where(slp[start:start+peak_index, i] <= thresh * peak_value)[0] #TODO test this '+start' addition
        if onset.size > 0:
            onsets[i] = start + onset[-1]
        else:
            onsets[i] = start
    return onsets

def compute_offset(slp, start, end, thresh=0.1):
    ''' Computes and approximated seizure offset per channel by first computing
        the peak of the envelope and then finding the first point in the envelope
        after the peak that has less than 10% of the peak amplitude '''
    # start = int(ts_on * seeg_info['sfreq'])
    # end = slp.shape[0]  # int((ts_on + 138) * seeg_info['sfreq'])
    offsets = np.empty(shape=slp.shape[1], dtype=int)
    for i in range(slp.shape[1]):
        peak_index = np.argmax(slp[start:end, i])
        peak_value = np.max(slp[start:end, i])
        offset = np.where(slp[start+peak_index:end, i] <= thresh * peak_value)[0]
        if offset.size > 0:
            offsets[i] = start + peak_index + offset[0]
        else:
            offsets[i] = end
    return offsets

def compute_signal_power(y, ch_names, bad=None):
    # should be renamed to signal_variance
    # NOTE: signal variance = signal power - mean squared; signal variance = to signal power with its mean removed

    ''' Computes signal power for all electrodes in both empirical and simulated SEEG'''
    for ind in range(y.shape[0]):
        y[ind, :] = y[ind, :] - y[ind, :].mean()#y[ind, 0]
    snsr_pwr = (y ** 2).mean(axis=1) # NOTE This is the signal variance formula !
    # snsr_pwr = (y ** 2).sum(axis=1)
    if bad is not None:
        for ch in bad:
                snsr_pwr[ch_names.index(ch)] = 0
    # energy_ch = (snsr_pwr - np.min(snsr_pwr)) / (np.max(snsr_pwr) - np.min(snsr_pwr))
    energy_ch = snsr_pwr/np.sum(snsr_pwr)
    return energy_ch

def compute_PSD(y, fs=1024):
    ''' Computes the power spectral density for all SEEG channels and plots it '''
    import scipy.signal
    # f contains the frequency components
    # S is the PSD
    (f, S) = scipy.signal.periodogram(y, fs, scaling='density')
    plt.figure()
    for i in range(S.shape[0]):
        plt.semilogy(f, S[i, :])
    # plt.semilogy(f, S[108,:])
    plt.ylim([1e-7, 1e8])
    plt.xlim([0, 100])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

# def compute_EZ_PZ_channels(t_onsets, energy_ch, ch_names, t_ez_max, signal_power_min):
#     '''
#     Computes the EZ and PZ channels in the SEEG signal (empirical/simulated)
#     NOTE : The t_ez_max and signal_power_min thresholds need to be adjusted differently for emp vis simulated SEEGs
#     '''
#     onset_ch_idx = np.where(t_onsets < t_ez_max)[0]  # the rest is PZ/NZ
#     prop_ch_idx = np.where(t_ez_max <= t_onsets)[0]
#     EZ_channels = []
#     PZ_channels = []
#     for ch_idx in onset_ch_idx:
#         if energy_ch[ch_idx] >= signal_power_min:
#             EZ_channels.append(ch_names[ch_idx])
#     for ch_idx in prop_ch_idx:
#         if energy_ch[ch_idx] >= signal_power_min:
#             PZ_channels.append(ch_names[ch_idx])
#     return EZ_channels, PZ_channels
def compute_SO_SP_channels(t_onsets, t_SO_max, binary_sc, ch_names):
    '''
    Computes the SO (seizure onset) and SP (seizure propagation) channels in the SEEG signal (empirical/simulated)
    t_onsets : onset time of seizure per channel
    t_SO_max : max time for seizure onset (basically the border between SO and SP)
    binary_sc : binarized seizure channels (seizure channel / no seizure channel)
    ch_names : channel names

    NOTE : The t_SO_max threshold needs to be adjusted differently for emp vis simulated SEEGs
    '''
    onset_ch_idx = np.where(t_onsets < t_SO_max)[0]  # the rest is PZ/NZ
    prop_ch_idx = np.where(t_SO_max <= t_onsets)[0]
    EZ_channels = []
    PZ_channels = []
    for ch_idx in onset_ch_idx:
        if binary_sc[ch_idx] == 1:
            EZ_channels.append(ch_names[ch_idx])
    for ch_idx in prop_ch_idx:
        if binary_sc[ch_idx] == 1:
            PZ_channels.append(ch_names[ch_idx])
    return EZ_channels, PZ_channels

def compare_occurrence(channels_emp, channels_sim):
    ''' Compares the co-occurrence of EZ/PZ channels between the empirical and the simulated data
        Using the empirical data as reference. '''
    if len(channels_emp) == 0:
        if len(channels_sim) == 0:
            # TODO NOTE : this change has only been applied starting from ControlCohortStimLocation dataset !!!
            return -100        # treating this as a special case, where no propagation is observed in both emp and sim
        else:
            return 0           # propagation only present in sim case, therefore occurrence = 0
    else:
        occurrence = 0         # count number of SP channels in emp case that also are present in sim case (Careful)
        for ch in channels_emp:
            if ch in channels_sim:
                occurrence += 1
        return occurrence/len(channels_emp) * 100

def jaccard_similarity_coeff(channels_emp, channels_sim):
    ''' Compares the overlap between two binary sets of same length using the jaccard similarity coefficient
        J = M11/(M01 + M10 + M11) * 100  where
        M11: total nr of attributes where both sets have a value of 1
        M01: total nr of attributes where the first set has a value of 0 and the second a value of 1
        M10: total nr of attributes where the first set has a value of 1 and the second a value of 0
        M00: total nr of attributes where both sets have a value of 0 '''
    if len(channels_emp) == 0:
        if len(channels_sim) == 0:
            # TODO NOTE : this change has only been applied starting from ControlCohortStimLocation dataset !!!
            return -100         # treating this as a special case, where no propagation is observed in both emp and sim
        else:
            return 0           # propagation only present in sim case, therefore occurrence = 0
    else:
        m11 = 0                # apply the Jaccard index
        m01 = 0
        m10 = 0
        for ch in channels_emp:
            if ch in channels_sim:
                m11 += 1
            else:
                m10 += 1
        for ch in channels_sim:
            if ch not in channels_emp:
                m01 += 1
    return m11/(m01 + m10 + m11) * 100

def compute_overlap(ez_channels_slp_grouped, ez_channels_slp_grouped_sim, ch_names, save_path=None, plot=True):
    '''
    Computes binary overlap between simulated and empirical data
    the accordance between no-seizure and seizure channels (in %)
    '''
    binary_emp = np.zeros(shape=len(ch_names), dtype=int)
    binary_sim = np.zeros(shape=len(ch_names), dtype=int)
    for i in range(len(ch_names)):
        # if energy_ch_emp[i] > 0.5 * np.mean(energy_ch_emp) or sc_channels_slp_grouped[i] == 1:
        # if peak_vals_grouped[i] > 1.2*np.mean(peak_vals_grouped):
        if ez_channels_slp_grouped[i] >= 0.35:
            binary_emp[i] = 1
        # if energy_ch_sim[i] > 0.05 * np.mean(energy_ch_sim) or sc_channels_slp_grouped_sim[i] == 1:
        # if peak_vals_grouped_sim[i] > np.mean(peak_vals_grouped_sim):
        if ez_channels_slp_grouped_sim[i] >= 0.35:
            binary_sim[i] = 1
    # computing the overlap
    overlap = (np.where(binary_sim + binary_emp == 2)[0].shape[0] +
               np.where(binary_sim + binary_emp == 0)[0].shape[0]) / len(ch_names)

    if plot:
        plt.figure(figsize=(25, 8))
        plt.bar(np.r_[1:len(ch_names) + 1], binary_emp, color='cornflowerblue', edgecolor='black', linewidth=2, alpha=0.5, log=False, label='Emp seizure', hatch='/')
        plt.bar(np.r_[1:len(ch_names) + 1], binary_sim, color='violet', edgecolor='black', linewidth=2, alpha=0.5, log=False, label='Sim seizure', hatch="\\")
        plt.xticks(np.r_[1:len(ch_names) + 1], ch_names, fontsize=17, rotation=45)
        plt.xlim([0, len(ch_names) + 1])
        plt.title(f'Overlap between empirical and simulated SEEG = {round(overlap, 2)}', fontsize=30)
        plt.legend(loc='upper right', fontsize=25)
        plt.ylabel('Binary value', fontsize=30)
        plt.xlabel('Electrodes', fontsize=30)
        plt.tight_layout()
        if save_path is not None:
            print('>> Save', f'{save_path}')
            plt.savefig(f'{save_path}')
        plt.show()
    return binary_emp, binary_sim, overlap

def compute_distance_matrix(electrodes_bip_xyz, ch_names_sim, plot=True):
    ''' Used to re-arrange the power spectrum measurements by taking into account
    mixing effects caused by neighboring electrodes observing the same signal twice
    IMPORTANT NOTE: not sure how useful this actually is... '''
    #Compute similarity for each electrode weighted by distance
    dist_matrix = np.empty([len(ch_names_sim), len(ch_names_sim)])
    for i, choi in enumerate(ch_names_sim):
        coord_choi = electrodes_bip_xyz[i]
        for j in range(len(ch_names_sim)):
            dist_matrix[i][j] = np.linalg.norm(coord_choi - electrodes_bip_xyz[j])
            dist_matrix[j][i] = dist_matrix[i][j]
    # Classify distance matrix into 3 main categories (for weighted distance)
    weight_dist_matrix = np.empty([len(ch_names_sim), len(ch_names_sim)])
    for i in range(len(ch_names_sim)):
        for j in range(len(ch_names_sim)):
            dist = dist_matrix[i][j]
            if dist == 0:
                weight = 1
            elif dist <= 3.5:
                weight = 0.5
            elif 3.5 < dist < 6.8:
                weight = 0.2
            else:
                weight = 0
            weight_dist_matrix[i][j] = weight
    if plot:
        plt.figure()
        plt.matshow(dist_matrix)
        plt.colorbar()
        plt.title('Distance matrix [mm]')
        plt.xlabel('Electrodes')
        plt.ylabel('Electrodes')
        plt.show()
        plt.figure()
        plt.matshow(weight_dist_matrix)
        plt.colorbar()
        plt.title('Weight distance matrix [mm]')
        plt.xlabel('Electrodes')
        plt.ylabel('Electrodes')
        plt.show()
    return dist_matrix, weight_dist_matrix

# def compute_NZ_channels(ch_names, EZ_channels, PZ_channels):
#     NZ_channels = []
#     for ch in ch_names:
#         if ch not in EZ_channels and ch not in PZ_channels:
#             NZ_channels.append(ch)
#     return NZ_channels

def get_electrodes(ch_names):
    '''From the list of bipolar channel names yields a list of electrode names
       e.g. from A'1-2 yields A' '''
    electrodes = []
    for ch in ch_names:
        num_digits = len(''.join(filter(str.isdigit, ch.split('-')[0])))
        electrodes.append(ch.split('-')[0][:-num_digits])
    return set(electrodes)

def get_channels(electrode, ch_names):
    '''For a particular electrode gives all channels corresponding to it
    e.g for TB' yields ["TB'1-2", "TB'2-3", "TB'3-4", "TB'6-7", "TB'7-8", "TB'8-9"] '''
    channels = \
    [ch for ch in ch_names if electrode == ch.split('-')[0][:-len(''.join(filter(str.isdigit, ch.split('-')[0])))]]
    return channels

def combine_channels(channels):
    ''' From a list of channels ASSUMING they come from the same electrode
        yields a unique channel containing the number of the first channel and the last one
        e.g. from [A'1-2, A'2-3, A'3-4] yields A'1-4'''
    num_digits = len(''.join(filter(str.isdigit, channels[0].split('-')[0])))
    electrode_name = channels[0].split('-')[0][:-num_digits]
    first_numbers = []
    for ch in channels:
        first_numbers.append(int(''.join(filter(str.isdigit, ch.split('-')[0]))))
    return f'{electrode_name}{min(first_numbers)}-{max(first_numbers)+1}'

def group_channels(ch_names, gain_prior):
    electrodes = get_electrodes(ch_names)
    channel_names_grouped_dict = {}
    for el in electrodes:
        # get list of channels for that electrode (sorry for ulgy code I wanted to put in one line)
        ch_list = np.asarray(get_channels(el, ch_names))
            #np.asarray([ch for ch in ch_names if el in ch])
        # get list of regions where each channel most strongly maps to
        region_list = np.asarray([np.argmax(gain_prior[ch_names.index(ch)]) for ch in ch_list])
        for region_idx in set(region_list):
            idxs = np.where(region_list == region_idx)[0]
            if idxs.shape[0] > 1:
                channels = ch_list[idxs]
                new_channel = combine_channels(channels)
                channel_names_grouped_dict[new_channel] = channels
            else:
                channels = ch_list[idxs][0]
                new_channel = channels # new channel = old channel
                channel_names_grouped_dict[new_channel] = np.asarray([channels])
    return channel_names_grouped_dict

def group_signal_power_and_onset(channel_names_grouped_dict, ch_names, energy_ch, peak_vals, ez_channels_slp, t_onsets):
    ''' Groups measurements according to grouped channel names
        energy_ch : signal power of a bipolar sensor
        peak_vals : maximum value of the envelope during seizure event for a bipolar sensor
        t_onsets : onset of seizure event for a bipolar sensor
        ez_channels_slp : value in [0, 1] interval; 0: no seizure channel, 1 : seizure channel
    '''
    energy_ch_grouped = []
    peak_vals_grouped = []
    onsets_grouped = []
    ez_channels_slp_grouped = []
    for ch_group in channel_names_grouped_dict.keys():
        channels = channel_names_grouped_dict.get(ch_group)
        sum_energy_ch = 0
        sum_peak_vals = 0
        sum_onsets = 0
        sum_ez_channels_slp = 0
        for ch in channels:
            idx = ch_names.index(ch)
            sum_energy_ch += energy_ch[idx]
            sum_peak_vals += peak_vals[idx]
            sum_onsets += t_onsets[idx]
            sum_ez_channels_slp += ez_channels_slp[idx]
        energy_ch_grouped.append(sum_energy_ch/len(channels))
        peak_vals_grouped.append(sum_peak_vals/len(channels))
        onsets_grouped.append(sum_onsets/len(channels))
        ez_channels_slp_grouped.append(sum_ez_channels_slp/len(channels))
    # normalize the measurements if necessary
    return energy_ch_grouped/sum(energy_ch_grouped), peak_vals_grouped/sum(peak_vals), ez_channels_slp_grouped,\
           onsets_grouped

def plot_SO_SP(channel_names_grouped, EZ_channels_grouped, PZ_channels_grouped, x_label='Empirical'):
    ez_val = np.zeros(shape=len(channel_names_grouped), dtype=int)
    pz_val = np.zeros(shape=len(channel_names_grouped), dtype=int)
    for i, ch in enumerate(channel_names_grouped):
        if ch in EZ_channels_grouped:
            ez_val[i] = 1
        elif ch in PZ_channels_grouped:
            pz_val[i] = 1
    plt.barh(np.r_[1:len(channel_names_grouped) + 1], ez_val, color='b', label='SO', edgecolor='k', alpha=0.4)
    plt.barh(np.r_[1:len(channel_names_grouped) + 1], pz_val, color='c', label='SP', edgecolor='k', alpha=0.4)
    # plt.ylabel('Electrodes', fontsize=36)
    plt.yticks(np.r_[1:len(channel_names_grouped) + 1], channel_names_grouped, fontsize=26)
    plt.ylim([0, len(channel_names_grouped) + 1])
    plt.xticks([])
    plt.xlabel(x_label, fontsize=36)
    plt.tight_layout()
    plt.legend(fontsize=24)

def plot_datafeatures(slp, save_path=None):
    plt.figure()
    plt.plot(slp)
    plt.xlabel('Time')
    plt.ylabel('Datafeatures')
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def plot_PC_correlation_matrix(correlation_matrix, n_comp, n_comp_sim, save_path=None):
    plt.figure()
    plt.imshow(correlation_matrix, label='Pearson')
    plt.colorbar()
    plt.title('Correlation matrix')
    plt.xticks(np.arange(n_comp_sim), np.arange(n_comp_sim)+1)
    plt.xlabel('PC simulated data')
    plt.yticks(np.arange(n_comp), np.arange(n_comp)+1)
    plt.ylabel('PC empirical data')
    plt.tight_layout()
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def plot_PCA_emp_sim(i, j, max_corr_val, PCA_emp, PCA_sim, ch_names_sim, comp_variance, comp_sim_variance, save_path=None):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10,8))
    plt.subplot(gs[0, :])
    plt.plot(PCA_emp, 'indigo', label=f'PC{i+1} Empirical', linewidth=2)
    plt.plot(PCA_sim, 'darkcyan', label=f'PC{j+1} Simulated', linewidth=2)
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=7, rotation=90)
    plt.xlim([0, len(ch_names_sim) + 1])
    plt.xlabel('Channels', fontsize=16)
    plt.ylabel('Principal components', fontsize=16)
    plt.title(f'Correlation: {round(max_corr_val, 2)}', fontsize=20)
    # plt.title(f'Correlation: {round(calc_correlation(PCA_emp/sum(PCA_emp), PCA_sim/sum(PCA_sim)) ,2)}', fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.subplot(gs[1, 0])
    plt.plot(comp_variance, '.', color='indigo')
    plt.title('Empirical', fontsize=16)
    plt.xlabel('Principal components', fontsize=16)
    plt.ylabel('Explained variance ratio', fontsize=16)
    plt.tight_layout()
    plt.subplot(gs[1, 1])
    plt.plot(comp_sim_variance, '.', color='darkcyan')
    plt.title('Simulated', fontsize=16)
    plt.xlabel('Principal components', fontsize=16)
    plt.ylabel('Explained variance ratio', fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def plot_binarized_SEEG(t, ch_names, onsets, offsets, ez_channels, seeg_info=None, save_path=None):
    fig = plt.figure(figsize=(40, 80))
    for ind, ich in enumerate(ch_names):
        if ez_channels[ind] == 1:
            plt.hlines(ind, t[onsets[ind]], t[offsets[ind]], color='black', linewidth=30)
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
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def plot_2d_binary_SEEG(binarized2Dslp_sim_new, ch_names_common, scaleplt=0.1, fig_size=(20, 45), marker_size=80, save_path = None):
    colors = ['white', 'black']
    levels = [0, 1]
    cmap, norm = mcolors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
    n_timepoints = binarized2Dslp_sim_new.shape[0]
    n_channels = binarized2Dslp_sim_new.shape[1]
    plt.figure(tight_layout=True, figsize=fig_size)
    for i in range(n_channels):
        plt.scatter(np.arange(n_timepoints), scaleplt*binarized2Dslp_sim_new[:, i] + i, c=binarized2Dslp_sim_new[:, i],
                    marker='s', cmap=cmap, norm=norm, s=marker_size, edgecolors=None)
    plt.yticks(np.arange(n_channels), ch_names_common, fontsize=70)
    plt.ylim(-1, n_channels) # n_channels+1
    plt.xlim(0, n_timepoints)
    plt.xticks([],fontsize=80)
    plt.xlabel('Time', fontsize=80)
    plt.ylabel('Electrodes', fontsize=80)
    plt.title('SEEG binarized', fontweight='bold', fontsize=80)
    if save_path is not None:
        print('>> Save', f'{save_path}')
        plt.savefig(f'{save_path}')
    plt.show()

def compute_stats(df, hyp='VEPhypothesis', group=None, plot=False):
    # compute stats of VEP/clinical Hypothesis for ALL patients
    corr_env_list = []
    corr_signal_pow_list = []
    binary_overlap_list = []

    SO_overlap_list = []
    SP_overlap_list = []
    NS_overlap_list = []
    J_SO_overlap_list = []
    J_SP_overlap_list = []
    J_NS_overlap_list = []

    PCA_correlation_list = []
    PCA1_correlation_list = []
    PCA_correlation_slp_list = []
    PCA1_correlation_slp_list = []

    agreement_img_list = []
    correlation_img_list = []
    mse_img_list = []
    rmse_img_list = []

    for sub in set(df['subject_id']):
        # select rows corresponding to a specific subject
        for row_id in df.loc[df['subject_id'] == sub].index:
            # select values from a specific EZ hypothesis
            if hyp in df['sim_seizure'][row_id]:
                # make sure the group field exists
                assert (group is None) or (group is not None and 'group' in df.keys())
                # if there's a group specified, select only values from that group
                if (group is None) or (group is not None and df['group'][row_id] == group):
                    corr_env = df['corr_envelope_amp'][row_id]
                    corr_signal_pow = df['corr_signal_pow'][row_id]
                    binary_overlap = df['binary_overlap'][row_id]

                    SO_overlap = df['SO_overlap'][row_id]
                    SP_overlap = df['SP_overlap'][row_id]
                    NS_overlap = df['NS_overlap'][row_id]

                    PCA_correlation = df['PCA_correlation'][row_id]
                    PCA1_correlation = df['PCA1_correlation'][row_id]
                    PCA_correlation_slp= df['PCA_correlation_slp'][row_id]
                    PCA1_correlation_slp = df['PCA1_correlation_slp'][row_id]

                    J_SO_overlap = df['J_SO_overlap'][row_id]
                    J_SP_overlap = df['J_SP_overlap'][row_id]
                    J_NS_overlap = df['J_NS_overlap'][row_id]

                    agreement_img = df['2D_agreement'][row_id]
                    correlation_img = df['2D_correlation'][row_id]
                    mse_img = df['2D_mse'][row_id]
                    rmse_img = df['2D_rmse'][row_id]

                    corr_env_list.append(corr_env)
                    corr_signal_pow_list.append(corr_signal_pow)
                    binary_overlap_list.append(binary_overlap)

                    SO_overlap_list.append(SO_overlap)
                    SP_overlap_list.append(SP_overlap)
                    NS_overlap_list.append(NS_overlap)
                    J_SO_overlap_list.append(J_SO_overlap)
                    J_SP_overlap_list.append(J_SP_overlap)
                    J_NS_overlap_list.append(J_NS_overlap)

                    PCA_correlation_list.append(PCA_correlation)
                    PCA1_correlation_list.append(PCA1_correlation)
                    PCA_correlation_slp_list.append(PCA_correlation_slp)
                    PCA1_correlation_slp_list.append(PCA1_correlation_slp)
                    agreement_img_list.append(agreement_img)
                    correlation_img_list.append(correlation_img)
                    mse_img_list.append(mse_img)
                    rmse_img_list.append(rmse_img)
                    if plot:
                        gs = gridspec.GridSpec(1, 2)
                        fig = plt.figure()
                        fig.suptitle(f'Subject {sub} ({hyp})')
                        plt.subplot(gs[0, 0])
                        plt.bar([0, 1, 2, 3], [binary_overlap * 100, SO_overlap, SP_overlap, NS_overlap])
                        plt.xticks([0, 1, 2, 3], ['Binary', 'SO', 'SP', 'NS'])
                        plt.ylim([0, 100 + 1])
                        plt.ylabel('Emp/Sim Overlap')
                        plt.tight_layout()

                        plt.subplot(gs[0, 1])
                        # plt.title(f'Subject {sub}')
                        plt.bar([0, 1, 2], [corr_env, corr_signal_pow, PCA_correlation], color='lightblue')
                        plt.xticks([0, 1, 2], ['Env. amp.', 'Sig. power', 'PCA comp.'])
                        plt.ylabel('Emp/Sim Correlation')
                        plt.ylim([0, 1])
                        plt.tight_layout()
                        plt.show()
    return binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
           J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
           PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list,\
           agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list

def plot_avg_stats_overlap(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list,
           J_SP_overlap_list, J_NS_overlap_list, hyp=None):
    fig = plt.figure()
    plt.title(f'{hyp}', fontsize=18)
    plt.errorbar([0, 1, 2, 3, 4, 5, 6], [np.mean(binary_overlap_list) * 100, np.mean(SO_overlap_list), np.mean(SP_overlap_list),
        np.mean(NS_overlap_list), np.mean(J_SO_overlap_list), np.mean(J_SP_overlap_list), np.mean(J_NS_overlap_list)],
        [np.std(binary_overlap_list) * 100, np.std(SO_overlap_list), np.std(SP_overlap_list), np.std(NS_overlap_list),
        np.std(J_SO_overlap_list), np.std(J_SP_overlap_list), np.std(J_NS_overlap_list)], fmt='o', ecolor='lightblue',
        elinewidth=3, capsize=0)
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Binary', 'SO', 'SP', 'NS', 'J_SO', 'J_SP', 'J_NS'], fontsize=18)
    plt.ylim([0, 100 + 1])
    plt.ylabel('Emp/Sim Overlap', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_avg_stats_overlap_two_datasets(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list,
                                        J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, binary_overlap_listc,
                                        SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, J_SO_overlap_listc,
                                        J_SP_overlap_listc, J_NS_overlap_listc, hyp=None):
    '''Plots VEC dataset measurements against control dataset measurements'''

    fig, ax = plt.subplots()
    plt.title(f'{hyp}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar([0, 1, 2, 3, 4, 5, 6],
                [np.mean(binary_overlap_list) * 100, np.mean(SO_overlap_list), np.mean(SP_overlap_list),
                 np.mean(NS_overlap_list), np.mean(J_SO_overlap_list), np.mean(J_SP_overlap_list),
                 np.mean(J_NS_overlap_list)],
                [np.std(binary_overlap_list) * 100, np.std(SO_overlap_list), np.std(SP_overlap_list),
                 np.std(NS_overlap_list), np.std(J_SO_overlap_list), np.std(J_SP_overlap_list),
                 np.std(J_NS_overlap_list)], fmt='o',
                ecolor='lightblue', color='steelblue', elinewidth=3, capsize=0, transform=trans1, label='VEC')
    ax.errorbar([0, 1, 2, 3, 4, 5, 6],
                [np.mean(binary_overlap_listc) * 100, np.mean(SO_overlap_listc), np.mean(SP_overlap_listc),
                 np.mean(NS_overlap_listc), np.mean(J_SO_overlap_listc), np.mean(J_SP_overlap_listc),
                 np.mean(J_NS_overlap_listc)],
                [np.std(binary_overlap_listc) * 100, np.std(SO_overlap_listc), np.std(SP_overlap_listc),
                 np.std(NS_overlap_listc), np.std(J_SO_overlap_listc), np.std(J_SP_overlap_listc),
                 np.std(J_NS_overlap_listc)], fmt='o',
                ecolor='mistyrose', color='tomato', elinewidth=3, capsize=0, transform=trans2, label='Control')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Binary', 'SO', 'SP', 'NS', 'J_SO', 'J_SP', 'J_NS'], fontsize=18)
    plt.ylim([0, 100 + 1])
    plt.ylabel('Emp/Sim Overlap', fontsize=18)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_avg_stats_correlation(corr_env_list, corr_signal_pow_list,
        PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, hyp=None):
    fig = plt.figure()
    plt.title(f'{hyp}', fontsize=18)
    plt.errorbar([0, 1, 2, 3, 4, 5], [np.mean(corr_env_list), np.mean(corr_signal_pow_list), np.mean(PCA_correlation_list),
                np.mean(PCA1_correlation_list), np.mean(PCA_correlation_slp_list), np.mean(PCA1_correlation_slp_list)],
                [np.std(corr_env_list),np.std(corr_signal_pow_list), np.std(PCA_correlation_list), np.std(PCA1_correlation_list),
                np.std(PCA_correlation_slp_list), np.std(PCA1_correlation_slp_list)], fmt='o', ecolor='lightblue',
                elinewidth=3, capsize=0)
    plt.xticks([0, 1, 2, 3, 4, 5], ['EnvA', 'SigPow', 'PCA', 'PCA1', 'PCAe', 'PCA1e'], fontsize=18)
    plt.ylabel('Emp/Sim Correlation', fontsize=18)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

def plot_avg_stats_correlation_two_datasets(corr_env_list, corr_signal_pow_list, PCA_correlation_list,
    PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, corr_env_listc, corr_signal_pow_listc,
    PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc, hyp=None):
    '''Plots VEC dataset measurements against control dataset measurements'''

    fig, ax = plt.subplots()
    plt.title(f'{hyp}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar([0, 1, 2, 3, 4, 5], [np.mean(corr_env_list), np.mean(corr_signal_pow_list), np.mean(PCA_correlation_list),
                np.mean(PCA1_correlation_list), np.mean(PCA_correlation_slp_list), np.mean(PCA1_correlation_slp_list)],
                [np.std(corr_env_list),np.std(corr_signal_pow_list), np.std(PCA_correlation_list), np.std(PCA1_correlation_list),
                np.std(PCA_correlation_slp_list), np.std(PCA1_correlation_slp_list)], fmt='o',
                ecolor='lightblue', color='steelblue', elinewidth=3, capsize=0, transform=trans1, label='VEC')
    ax.errorbar([0, 1, 2, 3, 4, 5], [np.mean(corr_env_listc), np.mean(corr_signal_pow_listc), np.mean(PCA_correlation_listc),
                np.mean(PCA1_correlation_listc), np.mean(PCA_correlation_slp_listc), np.mean(PCA1_correlation_slp_listc)],
                [np.std(corr_env_listc),np.std(corr_signal_pow_listc), np.std(PCA_correlation_listc), np.std(PCA1_correlation_listc),
                np.std(PCA_correlation_slp_listc), np.std(PCA1_correlation_slp_listc)], fmt='o',
                ecolor='mistyrose', color='tomato', elinewidth=3, capsize=0, transform=trans2, label='Control')
    plt.xticks([0, 1, 2, 3, 4, 5], ['EnvA', 'SigPow', 'PCA', 'PCA1', 'PCAe', 'PCA1e'], fontsize=18)
    plt.ylabel('Emp/Sim Correlation', fontsize=18)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_avg_stats_binary_img(agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, hyp=None):
    fig = plt.figure()
    plt.title(f'{hyp}', fontsize=18)
    plt.errorbar([0, 1, 2, 3], [np.mean(agreement_img_list), np.mean(correlation_img_list), np.mean(mse_img_list),
                np.mean(rmse_img_list)], [np.std(agreement_img_list),np.std(correlation_img_list), np.std(mse_img_list),
                np.std(rmse_img_list)], fmt='o', ecolor='lightblue', elinewidth=3, capsize=0)
    plt.xticks([0, 1, 2, 3], ['Agreement', 'Correlation', 'MSE', 'RMSE'], fontsize=18)
    plt.ylabel('Emp/Sim 2D binary SEEG', fontsize=18)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

def plot_avg_stats_binary_img_two_datasets(agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list,
    agreement_img_listc, correlation_img_listc, mse_img_listc, rmse_img_listc, hyp=None):
    '''Plots VEC dataset measurements against control dataset measurements'''

    fig, ax = plt.subplots()
    plt.title(f'{hyp}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

    ax.errorbar([0, 1, 2, 3], [np.mean(agreement_img_list), np.mean(correlation_img_list), np.mean(mse_img_list),
                np.mean(rmse_img_list)], [np.std(agreement_img_list),np.std(correlation_img_list), np.std(mse_img_list),
                np.std(rmse_img_list)], fmt='o', ecolor='lightblue', color='steelblue', elinewidth=3, capsize=0, transform=trans1,
                label='VEC')
    ax.errorbar([0, 1, 2, 3], [np.mean(agreement_img_listc), np.mean(correlation_img_listc), np.mean(mse_img_listc),
                               np.mean(rmse_img_listc)],
                [np.std(agreement_img_listc), np.std(correlation_img_listc), np.std(mse_img_listc), np.std(rmse_img_listc)],
                 fmt='o', ecolor='mistyrose', color='tomato', elinewidth=3, capsize=0, transform=trans2, label='Control')
    plt.xticks([0, 1, 2, 3], ['Agreement', 'Correlation', 'MSE', 'RMSE'], fontsize=18)
    plt.ylabel('Emp/Sim 2D binary SEEG', fontsize=18)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_avg_stats(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, corr_env_list,
                      corr_signal_pow_list, PCA_correlation_list, agreement_img_list, correlation_img_list, hyp=None):
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure()
    fig.suptitle(f'Mean across subjects ({hyp})')
    plt.subplot(gs[0, 0])
    plt.errorbar([0, 1, 2, 3, 4], [np.mean(binary_overlap_list) * 100, np.mean(SO_overlap_list), np.mean(SP_overlap_list),
                                np.mean(NS_overlap_list), np.mean(agreement_img_list)*100], [np.std(binary_overlap_list) * 100, np.std(SO_overlap_list),
                                        np.std(SP_overlap_list), np.std(NS_overlap_list), np.std(agreement_img_list)*100], linestyle='None', marker='^')
    plt.xticks([0, 1, 2, 3, 4], ['Binary', 'SO', 'SP', 'NS', 'Agrmt.'])
    plt.ylim([0, 100 + 1])
    plt.ylabel('Emp/Sim Overlap')
    plt.tight_layout()

    plt.subplot(gs[0, 1])
    # plt.title(f'Subject {sub}')
    plt.errorbar([0, 1, 2, 3], [np.mean(corr_env_list), np.mean(corr_signal_pow_list), np.mean(PCA_correlation_list), np.mean(correlation_img_list)],
                 [np.std(corr_env_list),np.std(corr_signal_pow_list), np.std(PCA_correlation_list), np.std(correlation_img_list)],
                 color='lightblue', linestyle='None', marker='^')
    plt.xticks([0, 1, 2, 3], ['Env. amp.', 'Sig. power', 'PCA comp.', 'Img.corr.'])
    plt.ylabel('Emp/Sim Correlation')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

