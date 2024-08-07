import sys
import matplotlib.pyplot as plt
import mne
import numpy as np
from src.utils import *
sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret, vep_prepare

#%% Load patient data, compute envelope and seizure onset for each channel
pid = 'id001_bt'  # 'id005_ft'#'id003_mg' # TODO change
szr_type = 'spontaneous' # TODO change
szr_names = load_seizure_name(pid, type=szr_type)
print(szr_names)
szr_index = 1         # TODO change
pid_bids = f'sub-{pid[2:5]}'
subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'
# Load seizure data
szr_name = f"{subjects_dir}/seeg/fif/{szr_names[szr_index].strip(' ')[1:-1]}.json"
bad = []
if szr_name == f"{subjects_dir}/seeg/fif/BTcrisePavecGeneralisation_0007.json":
    bad = ["H'8-9"]
seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subjects_dir, szr_name)
# load electrode positions
ch_names = bip.ch_names
# plot ts_on sec before and after
base_length = 10
start_idx = int((seeg_info['onset'] - base_length) * seeg_info['sfreq'])
end_idx = int((seeg_info['offset'] + base_length) * seeg_info['sfreq'])
y = bip.get_data()[:, start_idx:end_idx]
t = bip.times[start_idx:end_idx]
# plot_signal(t, y, ch_names, seeg_infoeg_info=seeg_info)
# compute enveloppe
hpf = 15
lpf = 0.04 # TODO change
ts_on = base_length
ts_off = base_length
ts_cut = ts_on/4
# ts_off_cut = seeg_info['offset'] + ts_off
slp = vep_prepare.compute_slp(seeg_info, bip, hpf=hpf, lpf=lpf, ts_on=ts_on, ts_off=ts_off)
expected_shape = round((seeg_info['offset'] - seeg_info['onset'] + ts_on + ts_off) * seeg_info['sfreq'])
assert np.fabs(slp.shape[0] - expected_shape) < 5, f'Expected shape: {expected_shape}, got {slp.shape[0]}'
# cut_off_N = int(ts_cut * seeg_info['sfreq'])
# ts_off_cut_N = int((ts_off_cut - seeg_info['onset'] + ts_on) * seeg_info['sfreq'])
# slp = slp[cut_off_N:ts_off_cut_N, :]
# expected_shape = round((ts_off_cut - seeg_info['onset'] + ts_on - ts_cut) * seeg_info['sfreq'])
# assert np.fabs(slp.shape[0] - expected_shape) < 5, f'Expected shape: {expected_shape}, got {slp.shape[0]}'
removebaseline = True
if removebaseline:
    baseline_N = int((ts_on - ts_cut) * seeg_info['sfreq'])
    slp = slp - np.mean(slp[:baseline_N, :], axis=0)
# Compute seizure onset for each channel
start = int(ts_on * seeg_info['sfreq'])
end = slp.shape[0] #int((ts_on + 140) * seeg_info['sfreq'])
onsets = compute_onset(slp, start, end, thresh=0.01)
plot_signal(t, y, ch_names, seeg_info=seeg_info, datafeature=slp, onsets=t[onsets])


#%% Do the same for synthetic data
clinical_hypothesis = False # TODO change
database = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'
sim_patient_data = f'{database}/{pid_bids}'
run = szr_index + 1
ses, task = get_ses_and_task(type='spontaneous')
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
# plot_signal(t_sim, y_sim_AC, ch_names_sim, scaleplt=0.1)
# compute enveloppe
hpf = 300 # TODO change
lpf = 10  # TODO change
ts_on = base_length
ts_off = base_length
ts_cut = ts_on/4
# ts_off_cut = seeg_info['offset'] + ts_off
slp_sim = compute_slp_sim(y_sim_AC.T, hpf=hpf, lpf=lpf, sfreq=1000)
# expected_shape = round((seeg_info['offset'] - seeg_info['onset'] + ts_on + ts_off) * seeg_info['sfreq'])
# assert np.fabs(slp.shape[0] - expected_shape) < 5, f'Expected shape: {expected_shape}, got {slp.shape[0]}'
# cut_off_N = int(ts_cut * seeg_info['sfreq'])
# ts_off_cut_N = int((ts_off_cut - seeg_info['onset'] + ts_on) * seeg_info['sfreq'])
# slp = slp[cut_off_N:ts_off_cut_N, :]
# expected_shape = round((ts_off_cut - seeg_info['onset'] + ts_on - ts_cut) * seeg_info['sfreq'])
# assert np.fabs(slp.shape[0] - expected_shape) < 5, f'Expected shape: {expected_shape}, got {slp.shape[0]}'
removebaseline = False
if removebaseline:
    baseline_N = 100#int((ts_on - ts_cut) * seeg_info['sfreq'])
    slp_sim = slp_sim - np.mean(slp[:100, :], axis=0)
# Compute onset for each channel
start_sim = 485#0
end_sim = slp_sim.shape[0]
onsets_sim = compute_onset(slp_sim, start_sim, end_sim)
plot_signal(t_sim, y_sim_AC, ch_names_sim, datafeature=slp_sim, onsets=t_sim[onsets_sim], scaleplt=0.1)


#%% Compare spatio-temporal features
# group electrodes in EZ, PZ electrodes (or also PZ1, PZ2).
# computing signal power and overlap between empirical and simulated
bad = ["OR'1-2"]
energy_ch = compute_signal_power(y, ch_names, bad)
# bad_sim = ["PI'1-2", "TB'", "A'", "B'", "H'", "I"]
bad_sim = ["PI'1-2"] + get_channels("TB'", ch_names_sim) +\
          get_channels("A'", ch_names_sim) + get_channels("B'", ch_names_sim) +\
          get_channels("H'", ch_names_sim) + get_channels("I", ch_names_sim) + get_channels("OR'", ch_names_sim) + \
          get_channels("GPH'", ch_names_sim)
energy_ch_sim = compute_signal_power(y_sim_AC, ch_names_sim, bad_sim)

# Empirical: group signals into EZ and PZ
t_ez_max = seeg_info['onset'] + 5
signal_power_min = 0.5 * np.mean(energy_ch)
EZ_channels, PZ_channels = compute_EZ_PZ_channels(t[onsets], energy_ch, ch_names, t_ez_max, signal_power_min)
NZ_channels = compute_NZ_channels(ch_names, EZ_channels, PZ_channels)
plot_signal_ez_pz(t, y, energy_ch, ch_names, EZ_channels, PZ_channels, seeg_info=seeg_info)

# Simulated: group signals into EZ and PZ
t_ez_max_sim = t_sim[onsets_sim].min() + 0.06
signal_power_min_sim = 0.1 * np.mean(energy_ch_sim) # used to be 0.1
EZ_channels_sim, PZ_channels_sim = compute_EZ_PZ_channels(t_sim[onsets_sim], energy_ch_sim, ch_names_sim,
                                                          t_ez_max_sim, signal_power_min_sim)
NZ_channels_sim = compute_NZ_channels(ch_names_sim, EZ_channels_sim, PZ_channels_sim)
plot_signal_ez_pz(t_sim, y_sim_AC, energy_ch_sim, ch_names_sim, EZ_channels_sim, PZ_channels_sim, scaleplt=0.1)
onset_ch_idx = np.where(t_sim[onsets_sim] < t_ez_max_sim)[0]
# compare the two EZ and PZ channels : how much is the co-occurrence between empirical and simulated SEEG
occurrence_ez = compare_occurrence(EZ_channels, EZ_channels_sim)
occurrence_pz = compare_occurrence(PZ_channels, PZ_channels_sim)
occurrence_nz = compare_occurrence(NZ_channels, NZ_channels_sim)
print(f'Occurrence ez {occurrence_ez}, Occurrence pz {occurrence_pz}, Occurrence nz {occurrence_nz}')


# computing the binary overlap (seizure/no seizure) between empirical and simulated SEEG
energy_ch_adjusted = []
for ch in ch_names_sim:
    idx = ch_names.index(ch)
    energy_ch_adjusted.append(energy_ch[idx])
overlap = compute_overlap(energy_ch_adjusted, energy_ch_sim, ch_names_sim, plot=True)
print(f'Overall overlap between empirical and simulated SEEG for pid {pid}: {overlap}')


#%% Group electrodes together following the gain matrix and compare groups

nch_sourse = []
for ind_ch, ichan in enumerate(ch_names_sim):
    isource = np.argmax(gain_prior[ind_ch])
    nch_sourse.append(f'{isource}:{ichan}')

def group_channels(ch_names, gain_prior):
    electrodes = get_electrodes(ch_names)
    channel_names_grouped = []
    channel_names_grouped_dict = {}
    for el in electrodes:
        # get list of channels for that electrode (sorry for ulgy code I wanted to put in one line)
        ch_list = np.asarray(get_channels(el))
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

            channel_names_grouped.append(new_channel)
    return channel_names_grouped, channel_names_grouped_dict

# TODO reminder: check first if both empirical and synthetic data have the same channels; if not modify it to be so !!!
channel_names_grouped, channel_names_grouped_dict = group_channels(ch_names, gain_prior)
# For all channel groups, compute average signal power and average onset time
def group_signal_power_and_onset(channel_names_grouped_dict, ch_names, energy_ch, t_onsets):
    energy_ch_grouped = []
    onsets_grouped = []
    for ch_group in channel_names_grouped_dict.keys():
        channels = channel_names_grouped_dict.get(ch_group)
        sum_energy_ch = 0
        sum_onsets = 0
        for ch in channels:
            idx = ch_names.index(ch)
            sum_energy_ch += energy_ch[idx]
            sum_onsets += t_onsets[idx]
        energy_ch_grouped.append(sum_energy_ch/len(channels))
        onsets_grouped.append(sum_onsets/len(channels))
        # normalize the measurements TODO
    return energy_ch_grouped, onsets_grouped

energy_ch_grouped, onsets_grouped = group_signal_power_and_onset(channel_names_grouped_dict, ch_names,
                                                                 energy_ch, t[onsets])

energy_ch_grouped_sim, onsets_grouped_sim = group_signal_power_and_onset(channel_names_grouped_dict, ch_names,
                                                                 energy_ch_sim, t_sim[onsets_sim])

plt.figure(figsize=(25, 8))
plt.bar(np.r_[1:len(energy_ch_grouped) + 1], energy_ch_grouped, color='blue', alpha=0.3, log=False, label='empirical')
plt.bar(np.r_[1:len(energy_ch_grouped_sim) + 1], energy_ch_grouped_sim, color='green', alpha=0.3, log=False, label='simulated')
plt.xticks(np.r_[1:len(channel_names_grouped) + 1], channel_names_grouped, fontsize=17, rotation=45)
plt.xlim([0, len(channel_names_grouped) + 1])
plt.legend(loc='upper right', fontsize=25)
plt.ylabel('Signal power', fontsize=25)
plt.xlabel('Electrodes', fontsize=25)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(energy_ch_grouped, energy_ch_grouped_sim, '.')
# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])
plt.show()

calc_correlation(energy_ch_grouped / sum(energy_ch_grouped), energy_ch_grouped_sim / sum(energy_ch_grouped_sim))


t_ez_max = seeg_info['onset'] + 5
signal_power_min = 0.5 * np.mean(energy_ch_grouped)
EZ_channels_grouped, PZ_channels_grouped = compute_EZ_PZ_channels(np.asarray(onsets_grouped), energy_ch_grouped,
                                                            channel_names_grouped, t_ez_max, signal_power_min)

t_ez_max_sim = min(onsets_grouped_sim) + 0.06
signal_power_min_sim = 1 * np.mean(energy_ch_grouped_sim)
EZ_channels_grouped_sim, PZ_channels_grouped_sim = compute_EZ_PZ_channels(onsets_grouped_sim, energy_ch_grouped_sim,
                                                          channel_names_grouped, t_ez_max_sim, signal_power_min_sim)
occurrence_ez = compare_occurrence(EZ_channels_grouped, EZ_channels_grouped_sim)
occurrence_pz = compare_occurrence(PZ_channels_grouped, PZ_channels_grouped_sim)
# occurrence_nz = compare_occurrence(NZ_channels, NZ_channels_sim)
print(f'Occurrence ez {occurrence_ez}, Occurrence pz {occurrence_pz},')


overlap = compute_overlap(energy_ch_grouped, energy_ch_grouped_sim, channel_names_grouped, plot=True)
print(f'Overall overlap between empirical and simulated SEEG for pid {pid}: {overlap}')
