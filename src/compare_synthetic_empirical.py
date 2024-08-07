'''
Applying measurements to compare the synthetic vs the empirical dataset
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
import pandas as pd
import sys
import mne
from src.utils import *
sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret


def plot_signal(t, y, energy_ch, ch_names, seeg_info=None, scaleplt=0.001):
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
    img = plt.barh(np.r_[1:energy_ch.shape[0] + 1], energy_ch, color='black', alpha=0.3, log=False)
    plt.ylabel('Electrodes', fontsize=50)
    plt.xlabel('Power', fontsize=50)
    plt.xticks(fontsize=26)
    plt.yticks(np.r_[1:len(ch_names) + 1], ch_names, fontsize=26)
    plt.ylim([0, len(ch_names) + 1])
    plt.title('SEEG channel power', fontweight='bold', fontsize=50)
    plt.tight_layout()
    plt.show()

def main():
    pid = 'id001_bt'  # 'id005_ft'#'id003_mg'
    pid_bids = f'sub-{pid[2:5]}'
    subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'
    # Load seizure data
    szr_name = f"{subjects_dir}/seeg/fif/BTcrisePavecGeneralisation_0007.json"
    seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subjects_dir, szr_name)
    # load electrode positions
    ch_names = bip.ch_names

    # plot ts_on sec before and after
    base_length = 5
    start_idx = int((seeg_info['onset'] - base_length) * seeg_info['sfreq'])
    base_length = 6
    end_idx = int((seeg_info['offset'] + base_length) * seeg_info['sfreq'])
    y = bip.get_data()[:, start_idx:end_idx]
    t = bip.times[start_idx:end_idx]
    bad = ["H'8-9"]
    energy_ch = compute_signal_power(y, ch_names, bad)
    # energy_ch[ch_names.index(bad[0])] = 0
    plot_signal(t, y, energy_ch, ch_names, seeg_info)

    # Load simulated seizure data
    database = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'
    sim_patient_data = f'{database}/{pid_bids}'
    run = 1
    ses = 1
    clinical_hypothesis = False
    if ses == 1:
        task = "simulatedseizure"
    elif ses == 2:
        task = "simulatedstimulation"
    elif ses == 3:
        task = 'simulatedinterictalspikes'
    if clinical_hypothesis:
        acq = "clinicalhypothesis"
    else:
        acq = "VEPhypothesis"
    print('ses ' + str(ses) + ' ' + task + ' ' + acq + ' run', run)
    seizure_name = f'{sim_patient_data}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'
    raw = mne.io.read_raw_brainvision(seizure_name, preload=True)
    y_sim = raw._data
    y_sim_AC = highpass_filter(y_sim, 256)
    t_sim = raw.times
    ch_names_sim = raw.ch_names
    energy_ch_sim = compute_signal_power(y_sim_AC, ch_names_sim)
    plot_signal(t_sim, y_sim_AC, energy_ch_sim, ch_names_sim, scaleplt=0.1)

    comparison = []
    for ch in ch_names_sim:
        idx = ch_names.index(ch)
        comparison.append(energy_ch[idx])

    plt.figure()
    plt.plot(comparison, energy_ch_sim, '.')
    plt.xlim([0,0.3])
    plt.ylim([0,0.3])
    plt.show()

    plt.figure(figsize=(25, 8))
    plt.bar(np.r_[1:len(comparison) + 1], comparison, color='blue', alpha=0.3, log=False, label='empirical')
    plt.bar(np.r_[1:energy_ch_sim.shape[0] + 1], energy_ch_sim, color='green', alpha=0.3, log=False, label='simulated')
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=17, rotation=45)
    plt.xlim([0, len(ch_names_sim) + 1])
    plt.legend(loc='upper right', fontsize=25)
    plt.ylabel('Signal power', fontsize=25)
    plt.xlabel('Electrodes', fontsize=25)
    plt.tight_layout()
    plt.show()

    # Pearson correlation
    def calc_correlation(actual, predic):
        a_diff = actual - np.mean(actual)
        p_diff = predic - np.mean(predic)
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
        return numerator / denominator
    res = calc_correlation(comparison/sum(comparison), energy_ch_sim/energy_ch_sim.sum())

    def calc_euclidian_distance(actual, predic):
        diff1 = np.sum(actual-predic)
        diff2 = np.sum(actual[1:] - predic[:-1])
        return diff1-diff2*0.5
    calc_euclidian_distance(comparison, energy_ch_sim)

    # Function to calculate Chi-distance
    def chi2_distance(A, B):
        # compute the chi-squared distance using above formula
        chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                            for (a, b) in zip(A, B)])
        return chi
    chi2_distance(np.asarray(comparison), energy_ch_sim)

    import scipy
    scipy.stats.chisquare(comparison/sum(comparison)+0.0000001, energy_ch_sim/energy_ch_sim.sum()+0.0000001)


    # Electrode locations
    electrodes_dir = f'{sim_patient_data}/{pid_bids}_electrodes.tsv'
    electrodes_data = pd.read_csv(electrodes_dir, sep='\t')
    electrodes = list(electrodes_data['name'])
    electrodes_xyz = np.vstack((electrodes_data['x'], electrodes_data['y'], electrodes_data['z'])).T

    # Compute bipolar electrode location
    electrodes_bip_xyz = np.empty([len(ch_names_sim), 3], dtype=float)
    for i, choi in enumerate(ch_names_sim):
        ch1 = choi.split('-')[0]
        ch1_id = electrodes.index(ch1)
        choi_xyz = 0.5 * (electrodes_xyz[ch1_id] + electrodes_xyz[ch1_id+1])
        electrodes_bip_xyz[i] = choi_xyz

    # Compute similarity for each electrode weighted by distance
    dist_matrix = np.empty([len(ch_names_sim), len(ch_names_sim)])
    for i, choi in enumerate(ch_names_sim):
        coord_choi = electrodes_bip_xyz[i]
        for j in range(len(ch_names_sim)):
            dist_matrix[i][j] = np.linalg.norm(coord_choi - electrodes_bip_xyz[j])
            dist_matrix[j][i] = dist_matrix[i][j]

    plt.figure()
    plt.matshow(dist_matrix)
    plt.colorbar()
    plt.title('Distance matrix [mm]')
    plt.xlabel('Electrodes')
    plt.ylabel('Electrodes')
    plt.show()

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

    plt.figure()
    plt.matshow(weight_dist_matrix)
    plt.colorbar()
    plt.title('Weight distance matrix [mm]')
    plt.xlabel('Electrodes')
    plt.ylabel('Electrodes')
    plt.show()

    # choi = "TP'4-5"
    # choi_id = ch_names_sim.index(choi)
    difference = np.empty(shape=len(comparison))
    avg_emp = np.empty(shape=len(comparison))
    avg_sim = np.empty(shape=len(comparison))
    for choi_id in range(len(comparison)):
        val_emp = 0
        val_sim = 0
        N = 0
        for i, ch in enumerate(ch_names_sim):
            if weight_dist_matrix[choi_id][i] > 0:
                val_emp += comparison[i] * weight_dist_matrix[choi_id][i]
                val_sim += energy_ch_sim[i] * weight_dist_matrix[choi_id][i]
                N += weight_dist_matrix[choi_id][i]
            # difference += ((comparison[choi_id] - energy_ch_sim[i])**2) * weight_dist_matrix[choi_id][i]
        difference[choi_id] = abs(val_emp/N - val_sim/N)#abs(val_emp - val_sim) # abs(val_emp - val_sim)*100/(N*val_emp)
        avg_emp[choi_id] = val_emp/N
        avg_sim[choi_id] = val_sim/N


    plt.figure(figsize=(25, 8))
    plt.bar(np.r_[1:len(difference) + 1], difference, color='blue', alpha=0.3, log=False, label='difference in %')
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=17, rotation=45)
    plt.xlim([0, len(ch_names_sim) + 1])
    # plt.ylim([0,300])
    plt.hlines(0.01, xmin=0, xmax=len(ch_names_sim) + 1)
    plt.legend(loc='upper right', fontsize=25)
    plt.ylabel('Signal power', fontsize=25)
    plt.xlabel('Electrodes', fontsize=25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(25, 8))
    plt.bar(np.r_[1:len(comparison) + 1], abs(comparison - energy_ch_sim), color='blue', alpha=0.3, log=False, label='difference')
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=17, rotation=45)
    plt.xlim([0, len(ch_names_sim) + 1])
    plt.legend(loc='upper right', fontsize=25)
    plt.ylabel('Signal power', fontsize=25)
    plt.xlabel('Electrodes', fontsize=25)
    plt.tight_layout()
    plt.show()

    # doing a binary comparison
    plt.figure(figsize=(25, 8))
    plt.bar(np.r_[1:len(comparison) + 1], avg_emp, color='blue', alpha=0.3, log=False, label='empirical')
    plt.bar(np.r_[1:len(comparison) + 1], avg_sim, color='green', alpha=0.3, log=False, label='simulated')
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=17, rotation=45)
    plt.xlim([0, len(ch_names_sim) + 1])
    plt.hlines(0.1*np.mean(avg_emp), xmin=0, xmax=len(ch_names_sim) + 1)
    # plt.hlines(0.1*np.mean(avg_sim), xmin=0, xmax=len(ch_names_sim) + 1)
    plt.legend(loc='upper right', fontsize=25)
    plt.ylabel('Signal power', fontsize=25)
    plt.xlabel('Electrodes', fontsize=25)
    plt.tight_layout()
    plt.show()

    binary_sim = np.zeros(shape=len(ch_names_sim), dtype=int)
    binary_emp = np.zeros(shape=len(ch_names_sim), dtype=int)
    for i in range(len(ch_names_sim)):
        if avg_emp[i] > 0.5*np.mean(avg_emp):
            binary_emp[i] = 1
        if avg_sim[i] > 0.05*np.mean(avg_sim):
            binary_sim[i] = 1

    plt.figure(figsize=(25, 8))
    plt.bar(np.r_[1:len(comparison) + 1], binary_emp, color='blue', alpha=0.3, log=False, label='empirical')
    plt.bar(np.r_[1:len(comparison) + 1], binary_sim, color='green', alpha=0.3, log=False, label='simulated')
    plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=17, rotation=45)
    plt.xlim([0, len(ch_names_sim) + 1])
    plt.legend(loc='upper right', fontsize=25)
    plt.ylabel('Binary power', fontsize=25)
    plt.xlabel('Electrodes', fontsize=25)
    plt.tight_layout()
    plt.show()

    # computing the overlap
    (np.where(binary_sim+binary_emp == 2)[0].shape[0] + np.where(binary_sim+binary_emp == 0)[0].shape[0])/len(comparison)

    binary_sim = np.zeros(shape=len(ch_names_sim), dtype=int)
    binary_emp = np.zeros(shape=len(ch_names_sim), dtype=int)
    for i in range(len(ch_names_sim)):
        if comparison[i] > 0.5*np.mean(comparison):
            binary_emp[i] = 1
        if energy_ch_sim[i] > 0.05*np.mean(energy_ch_sim):
            binary_sim[i] = 1


