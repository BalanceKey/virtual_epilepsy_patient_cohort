import os
import mne
from scipy.stats import pearsonr
from scipy import stats
from sewar.full_ref import mse, rmse
from src.utils import *
from src.utils_functions import vep_prepare_ret


#%% Load patient data, compute envelope and seizure onset for each channel
pid = 'id030_bf' # TODO change
szr_index = 1       # TODO change
clinical_hypothesis = False # TODO change

szr_type = 'spontaneous' # TODO change
bad = [] # TODO change
bad_sim = []#["B8-9", "B7-8", "OT1-2", "OT2-3"] # TODO change
hpf = 20
lpf = 0.15#0.02 # TODO change
base_length = 5 # TODO change
filt_order = 156#256 # TODO change
hpf_sim = 300 # TODO change
lpf_sim = 10#8  # TODO change
sfreq_sim = 1000 # TODO change
base_length_sim = 100 # TODO change

szr_names = load_seizure_name(pid, type=szr_type)
print(szr_names)
pid_bids = f'sub-{pid[2:5]}'
subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'
save_analysis = f'{subjects_dir}/comparison_emp_sim' # To save the stats
os.makedirs(save_analysis, exist_ok=True)

# Load seizure data
szr_name = f"{subjects_dir}/seeg/fif/{szr_names[szr_index].strip(' ')[1:-1]}.json"
# szr_name = f"{subjects_dir}/seeg/fif/{szr_names[szr_index].strip(' ')}.json"
seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subjects_dir, szr_name)
# load electrode positions
ch_names = bip.ch_names
# plot ts_on sec before and after
start_idx = int((seeg_info['onset'] - base_length) * seeg_info['sfreq'])
end_idx = int((seeg_info['offset'] + base_length) * seeg_info['sfreq'])
y = bip.get_data()[:, start_idx:end_idx]
t = bip.times[start_idx:end_idx]
# compute enveloppe
# hpf = 30
# lpf = 0.04
ts_on = base_length
ts_off = base_length
ts_cut = ts_on/4
# ts_off_cut = seeg_info['offset'] + ts_off
slp = vep_prepare_ret.compute_slp(seeg_info, bip, hpf=hpf, lpf=lpf, ts_on=ts_on, ts_off=ts_off)
expected_shape = round((seeg_info['offset'] - seeg_info['onset'] + ts_on + ts_off) * seeg_info['sfreq'])
assert np.fabs(slp.shape[0] - expected_shape) < 5, f'Expected shape: {expected_shape}, got {slp.shape[0]}'
removebaseline = True
if removebaseline:
    baseline_N = int((ts_on - ts_cut) * seeg_info['sfreq'])
    slp = slp - np.mean(slp[:baseline_N, :], axis=0)
# Compute seizure onset for each channel
start = int(ts_on * seeg_info['sfreq'])
end = int((ts_on + seeg_info['offset']-seeg_info['onset']) * seeg_info['sfreq'])#slp.shape[0] #int((ts_on + 140) * seeg_info['sfreq'])
onsets = compute_onset(slp, start, end, thresh=0.01)
offsets = compute_offset(slp, start, end, thresh=0.0001)
# plot datafeatures
plot_datafeatures(slp, save_path=f"{save_analysis}/{szr_names[szr_index].strip(' ')[1:-1]}_datafeatures.png")

# compute from datafeatures which channels are SC (seizure channel) and which are NSC (no seizure channel)
# peak_idx = np.argmax(slp[start_idx:end_idx, :], axis=0)
base_length_sfreq = int(base_length*seeg_info['sfreq'])
peak_vals = np.max(slp[base_length_sfreq:-base_length_sfreq, :], axis=0)
ez_channels_slp = np.zeros(shape = len(ch_names), dtype='int')
ezthresh = slp.mean() + 1.1 * slp.std() # 4.5 # TODO change
print(ezthresh)
for ich, ch in enumerate(ch_names):
    if peak_vals[ich] > ezthresh:
        ez_channels_slp[ich] = 1
if len(bad) > 0:
    for ch in bad:
        ez_channels_slp[ch_names.index(ch)] = 0

# plot seizure with datafeatures and onset times
save_path = f"{save_analysis}/{szr_names[szr_index].strip(' ')[1:-1]}.png"
plot_signal(t, y, ch_names, scaleplt=0.002, seeg_info=seeg_info, datafeature=slp, ez_channels_slp=ez_channels_slp,
            onsets=t[onsets], offsets=t[offsets], save_path=save_path)

#%% Do the same for synthetic data
# clinical_hypothesis = False
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
y_sim_AC = highpass_filter(y_sim, 256, filter_order = 101)
t_sim = raw.times
ch_names_sim = raw.ch_names
# compute enveloppe
# hpf = 300
# lpf = 4
# ts_on = base_length
# ts_off = base_length
# ts_cut = ts_on/4
slp_sim = compute_slp_sim(y_sim_AC.T, hpf=hpf_sim, lpf=lpf_sim, sfreq=1000)
removebaseline = True # TODO change
if removebaseline:
    baseline_N = 5#int((ts_on - ts_cut) * seeg_info['sfreq'])
    slp_sim = slp_sim - np.mean(slp_sim[:baseline_N, :], axis=0)
# Compute onset for each channel
start_sim = 0
end_sim = slp_sim.shape[0]-base_length_sim
onsets_sim = compute_onset(slp_sim, start_sim, end_sim, thresh=0.2) # 0.1 TODO thresh changed here is that ok?
offsets_sim = compute_offset(slp_sim, start_sim, end_sim, thresh=0.0001)
# plot datafeatures
plot_datafeatures(slp_sim, save_path=f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg_datafeatures.png")

# compute from datafeatures which channels are SC (seizure channel) and which are NSC (no seizure channel)
# peak_idx_sim = np.argmax(slp_sim[start_idx_sim:end_idx_sim, :], axis=0)
peak_vals_sim = np.max(slp_sim[base_length_sim:-base_length_sim, :], axis=0)
ez_channels_slp_sim = np.zeros(shape=len(ch_names_sim), dtype='int')
ezthresh_sim = slp_sim.mean() + 1.8 * slp_sim.std() # TODO change
print(ezthresh_sim)
for ich, ch in enumerate(ch_names_sim):
    if peak_vals_sim[ich] > ezthresh_sim:
        ez_channels_slp_sim[ich] = 1

# plot seizure with datafeatures and onset times
save_path_sim = f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.png"
plot_signal(t_sim, y_sim_AC, ch_names_sim, datafeature=slp_sim, ez_channels_slp=ez_channels_slp_sim,
            onsets=t_sim[onsets_sim], offsets=t_sim[offsets_sim], scaleplt=0.1, save_path=save_path_sim)

#%% COMPARE spatio-temporal features
# if assertion is false, adjust the channels and gain matrix !!
if szr_name == f"{subjects_dir}/seeg/fif/BTcrisePavecGeneralisation_0007.json":
    bad = ["H'8-9"]
elif szr_name == f"{subjects_dir}/seeg/fif/SDcrisettesinfracliniques_0000_4.json":
    bad = ["B1-2", "B2-3", "B'1-2", "B'2-3"]
elif szr_name == f"{subjects_dir}/seeg/fif/GJL_crise2_P_090318CBEX_0001.json":
    bad = ["OP2-3"]
energy_ch = compute_signal_power(y, ch_names, bad)
# bad_sim = []  # ["PI'1-2"] + get_channels("TB'", ch_names_sim) +\
# get_channels("A'", ch_names_sim) + get_channels("B'", ch_names_sim) +\
energy_ch_sim = compute_signal_power(y_sim_AC, ch_names_sim, bad_sim)

assert len(ch_names_sim) == len(ch_names)
assert ch_names == ch_names_sim

if len(ch_names_sim) == len(ch_names):
    ch_names_common = ch_names
    energy_ch_c = energy_ch
    energy_ch_sim_c = energy_ch_sim
    gain_prior_c = gain_prior
    peak_vals_c = peak_vals
    peak_vals_sim_c = peak_vals_sim
    ez_channels_slp_c = ez_channels_slp
    ez_channels_slp_sim_c = ez_channels_slp_sim
    y_c = y
    y_sim_AC_c = y_sim_AC
    onsets_c = onsets
    onsets_sim_c = onsets_sim
    offsets_c = offsets
    offsets_sim_c = offsets_sim
    slp_c = slp
    slp_sim_c = slp_sim
else:
    if len(ch_names_sim) > len(ch_names):
        ch_names_common = []
        for ch in ch_names_sim:
            if ch in ch_names:
                ch_names_common.append(ch)
    elif len(ch_names_sim) < len(ch_names):
        ch_names_common = []
        for ch in ch_names:
            if ch in ch_names_sim:
                ch_names_common.append(ch)
    else:
        print('ERROR NOT POSSIBLE !!!!!!!!')

    energy_ch_c = np.empty(shape=len(ch_names_common))
    energy_ch_sim_c = np.empty(shape=len(ch_names_common))
    peak_vals_c = np.empty(shape=len(ch_names_common))
    peak_vals_sim_c = np.empty(shape=len(ch_names_common))
    ez_channels_slp_c = np.empty(shape=len(ch_names_common))
    ez_channels_slp_sim_c = np.empty(shape=len(ch_names_common))
    gain_prior_c = np.empty(shape=(len(ch_names_common), 162))
    y_c = np.empty(shape=(len(ch_names_common), t.size))
    y_sim_AC_c = np.empty(shape=(len(ch_names_common), t_sim.size))
    onsets_c = np.empty(shape=len(ch_names_common), dtype=int)
    onsets_sim_c = np.empty(shape=len(ch_names_common), dtype=int)
    offsets_c = np.empty(shape=len(ch_names_common), dtype=int)
    offsets_sim_c = np.empty(shape=len(ch_names_common), dtype=int)
    slp_c = np.empty(shape=(t.size, len(ch_names_common)))
    slp_sim_c = np.empty(shape=(t_sim.size, len(ch_names_common)))
    for i, ch in enumerate(ch_names_common):
        idx = ch_names.index(ch)
        idx_sim = ch_names_sim.index(ch)
        energy_ch_c[i] = energy_ch[idx]
        energy_ch_sim_c[i] = energy_ch_sim[idx_sim]
        peak_vals_c[i] = peak_vals[idx]
        peak_vals_sim_c[i] = peak_vals_sim[idx_sim]
        ez_channels_slp_c[i] = ez_channels_slp[idx]
        ez_channels_slp_sim_c[i] = ez_channels_slp_sim[idx_sim]
        gain_prior_c[i] = gain_prior[idx]
        y_c[i] = y[idx]
        y_sim_AC_c[i] = y_sim_AC[idx_sim]
        onsets_c[i] = onsets[idx]
        onsets_sim_c[i] = onsets_sim[idx_sim]
        offsets_c[i] = offsets[idx]
        offsets_sim_c[i] = offsets_sim[idx_sim]
        slp_c[:, i] = slp[:, idx]
        slp_sim_c[:, i] = slp_sim[:, idx_sim]
channel_names_grouped_dict = group_channels(ch_names_common, gain_prior_c)
# For all channel groups, compute average signal power and average onset time
energy_ch_grouped, peak_vals_grouped, ez_channels_slp_grouped, onsets_grouped = \
        group_signal_power_and_onset(channel_names_grouped_dict, ch_names_common, energy_ch_c, peak_vals_c,
                                     ez_channels_slp_c, t[onsets_c])
energy_ch_grouped_sim, peak_vals_grouped_sim, ez_channels_slp_grouped_sim, onsets_grouped_sim =\
        group_signal_power_and_onset(channel_names_grouped_dict, ch_names_common, energy_ch_sim_c, peak_vals_sim_c,
                                     ez_channels_slp_sim_c, t_sim[onsets_sim_c])
channel_names_grouped = list(channel_names_grouped_dict.keys())
#%% FIGURE 1 : Binarizing the signal power between seizure and no seizure and comparing the empirical and simulated results
# For all channel groups, compute for datafeature (slp) if channel is seizure channel or not seizure channel
save_path_occurrence = f"{save_analysis}/Fig1_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_overlap_emp_sim.png"
binary_emp, binary_sim, overlap = compute_overlap(ez_channels_slp_grouped, ez_channels_slp_grouped_sim, channel_names_grouped, save_path=save_path_occurrence)
print(f'Overall overlap between empirical and simulated SEEG for pid {pid}: {overlap}')
#%%

save=True
gs = gridspec.GridSpec(2, 1)
plt.figure(figsize=(25, 16))
plt.subplot(gs[1, :])
plt.bar(np.r_[1:len(energy_ch_grouped) + 1], energy_ch_grouped, color='blue', alpha=0.3, log=False, label='empirical')
plt.bar(np.r_[1:len(energy_ch_grouped_sim) + 1], energy_ch_grouped_sim, color='green', alpha=0.3, log=False, label='simulated')
plt.xticks(np.r_[1:len(channel_names_grouped) + 1], channel_names_grouped, fontsize=17, rotation=45)
plt.xlim([0, len(channel_names_grouped) + 1])
plt.legend(loc='upper right', fontsize=28)
plt.ylabel('Signal power', fontsize=36)
# plt.xlabel('Electrodes', fontsize=28)
plt.tight_layout()
plt.subplot(gs[0, :])
plt.bar(np.r_[1:len(peak_vals_grouped) + 1], peak_vals_grouped, color='blue', alpha=0.3, log=False, label='empirical')
plt.bar(np.r_[1:len(peak_vals_grouped_sim) + 1], peak_vals_grouped_sim, color='green', alpha=0.3, log=False, label='simulated')
plt.xticks(np.r_[1:len(channel_names_grouped) + 1], channel_names_grouped, fontsize=17, rotation=45)
plt.xlim([0, len(channel_names_grouped) + 1])
plt.legend(loc='upper right', fontsize=28)
plt.ylabel('Envelope amplitude', fontsize=36)
plt.xlabel('Electrodes', fontsize=28)
plt.tight_layout()
if save:
    print('>> Save', f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_sigpower_emp_sim_grouped.png")
    plt.savefig(f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_sigpower_emp_sim_grouped.png")
plt.show()

gs = gridspec.GridSpec(1, 2)
plt.figure()
plt.subplot(gs[0, 1])
corr_signal_pow = calc_correlation(energy_ch_grouped, energy_ch_grouped_sim)
plt.plot(energy_ch_grouped, energy_ch_grouped_sim, '.', label=f'correlation={round(corr_signal_pow, 2)}')
plt.legend()
plt.xlabel('Grouped signal power empirical')
plt.ylabel('Grouped signal power simulated')
plt.tight_layout()
plt.subplot(gs[0, 0])
corr_envelope_amp = calc_correlation(peak_vals_grouped, peak_vals_grouped_sim)
plt.plot(peak_vals_grouped, peak_vals_grouped_sim, '.', label=f'correlation={round(corr_envelope_amp, 2)}')
plt.legend()
plt.xlabel('Grouped envelope amplitude empirical')
plt.ylabel('Grouped envelope amplitude simulated')
plt.tight_layout()
if save:
    print('>> Save', f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_emp_sim_correlation.png")
    plt.savefig(f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_emp_sim_correlation.png")
plt.show()
print('Correlation envelope amp = ', corr_envelope_amp, '\n Correlation signal power = ', corr_signal_pow)


t_SO_max = seeg_info['onset'] + 30# TODO see if needs change
EZ_channels_grouped, PZ_channels_grouped = compute_SO_SP_channels(np.asarray(onsets_grouped), t_SO_max, binary_emp,
                                                            list(channel_names_grouped))
NZ_channels_grouped = [channel_names_grouped[i] for i in np.where(binary_emp == 0)[0]]
# t_ez_max = seeg_info['onset'] + 5

EZ_channels = sum([list(channel_names_grouped_dict[ez]) for ez in EZ_channels_grouped], [])
PZ_channels = sum([list(channel_names_grouped_dict[pz]) for pz in PZ_channels_grouped], [])
save_path_ezpz = f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_emp_ez_pz.png"
plot_signal_ez_pz(t, y, peak_vals, ch_names, EZ_channels, PZ_channels, seeg_info=seeg_info, save_path=save_path_ezpz)

t_SO_max_sim = min([onsets_grouped_sim[i] for i in np.where(binary_sim == 1)[0]]) + 0.1#0.1 # TODO see if needs change
EZ_channels_grouped_sim, PZ_channels_grouped_sim = compute_SO_SP_channels(np.asarray(onsets_grouped_sim), t_SO_max_sim, binary_sim,
                                                            list(channel_names_grouped))
NZ_channels_grouped_sim = [channel_names_grouped[i] for i in np.where(binary_sim == 0)[0]]
EZ_channels_sim = sum([list(channel_names_grouped_dict[ez]) for ez in EZ_channels_grouped_sim], [])
PZ_channels_sim = sum([list(channel_names_grouped_dict[pz]) for pz in PZ_channels_grouped_sim], [])
save_path_ezpz_sim = f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_sim_ez_pz.png"
plot_signal_ez_pz(t_sim, y_sim_AC, peak_vals_sim, ch_names_sim, EZ_channels_sim, PZ_channels_sim, scaleplt=0.1, save_path=save_path_ezpz_sim)

#%% FIGURE 2
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
fig = plt.figure(figsize=(10, 20))
plt.subplot(gs[0])
plot_SO_SP(channel_names_grouped, EZ_channels_grouped, PZ_channels_grouped)
plt.ylabel('Electrodes', fontsize=36)
plt.subplot(gs[1])
plot_SO_SP(channel_names_grouped, EZ_channels_grouped_sim, PZ_channels_grouped_sim, x_label='Simulated')
save_path_SO_SP = f"{save_analysis}/Fig2_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_SO_SP.png"
print('>> Save', f'{save_path_SO_SP}')
plt.savefig(save_path_SO_SP)
plt.show()
occurrence_SO = compare_occurrence(EZ_channels_grouped, EZ_channels_grouped_sim)
occurrence_SP = compare_occurrence(PZ_channels_grouped, PZ_channels_grouped_sim)
occurrence_NS = compare_occurrence(NZ_channels_grouped, NZ_channels_grouped_sim)
print(f'Occurrence SO: {occurrence_SO}; Occurrence SP: {occurrence_SP}; Occurrence NS: {occurrence_NS}')
jaccard_coeff_SO = jaccard_similarity_coeff(EZ_channels_grouped, EZ_channels_grouped_sim)
jaccard_coeff_SP = jaccard_similarity_coeff(PZ_channels_grouped, PZ_channels_grouped_sim)
jaccard_coeff_NS = jaccard_similarity_coeff(NZ_channels_grouped, NZ_channels_grouped_sim)
print(f'Jacc Overlap SO: {jaccard_coeff_SO}; Jacc Overlap SP: {jaccard_coeff_SP}; Jacc Overlap NS: {jaccard_coeff_NS}')
#%%

#%% PCA
# # Standardize the data
# y_standard = (y.T - np.mean(y.T, axis=0)) / np.std(y.T, axis = 0)
# plot_signal(t, y_standard.T, ch_names, datafeature=None, onsets=None, scaleplt=0.2)
# Compute PCA on empirical timeseries
pca = get_PCA_components(y_c.T, ch_names_common)
comp = pca.components_
comp_variance = pca.explained_variance_ratio_
n_comp = min(10, feature_vector_dim(pca, max_variance=0.90))
# Compute PCA on simulated timeseries
pca_sim = get_PCA_components(y_sim_AC_c.T, ch_names_common)
comp_sim = pca_sim.components_
comp_sim_variance = pca_sim.explained_variance_ratio_
n_comp_sim = min(10, feature_vector_dim(pca_sim, max_variance=0.90))
# Compute correlation between main principal components (the ones that explain most of the variance)
correlation_matrix = np.empty(shape=(n_comp, n_comp_sim))
for i in range(n_comp):
    PCA_emp = comp[i]
    # multiply vector with -1 if max peak is negative
    if PCA_emp[np.argmax(np.absolute(PCA_emp))] < 0:
        PCA_emp = PCA_emp * -1
    for j in range(n_comp_sim):
        PCA_sim = comp_sim[j]
        if PCA_sim[np.argmax(np.absolute(PCA_sim))] < 0:
            PCA_sim = PCA_sim * -1
        correlation_matrix[i][j] = np.absolute(pearsonr(PCA_emp/np.sum(PCA_emp), PCA_sim/np.sum(PCA_sim))[0])
# Plot matrix
plot_PC_correlation_matrix(correlation_matrix, n_comp, n_comp_sim,
        save_path=f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_PCA_emp_sim_corr_matrix.png")
# Extract the PC_emp, PC_sim pair with the highest correlation and plot
i, j = np.unravel_index(np.argmax(correlation_matrix, axis=None), correlation_matrix.shape)
max_corr_val = np.max(correlation_matrix)
PCA_emp = comp[i]
PCA_sim = comp_sim[j]
# multiply vector with -1 if max peak is negative
if PCA_emp[np.argmax(np.absolute(PCA_emp))] < 0:
    PCA_emp = PCA_emp * -1
if PCA_sim[np.argmax(np.absolute(PCA_sim))] < 0:
    PCA_sim = PCA_sim * -1
plot_PCA_emp_sim(i, j, max_corr_val, PCA_emp, PCA_sim, ch_names_common, comp_variance, comp_sim_variance,
                 save_path=f"{save_analysis}/Fig3_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_PCA_emp_sim.png")
print('Correlation PCA1emp and PCA1sim = ', max_corr_val)#calc_correlation(PCA_emp, PCA_sim))
# Correlation between two first PCA components
PCA1_corr_val = correlation_matrix[0][0]

# SAME PCA ON SLP (DATAFEATURES)
pca = get_PCA_components(slp_c, ch_names_common)
comp = pca.components_
comp_variance = pca.explained_variance_ratio_
n_comp = min(10, feature_vector_dim(pca, max_variance=0.99))
# Compute PCA on simulated timeseries
pca_sim = get_PCA_components(slp_sim_c, ch_names_common)
comp_sim = pca_sim.components_
comp_sim_variance = pca_sim.explained_variance_ratio_
n_comp_sim = min(10, feature_vector_dim(pca_sim, max_variance=0.99))
# Compute correlation between main principal components (the ones that explain most of the variance)
correlation_matrix = np.empty(shape=(n_comp, n_comp_sim))
for i in range(n_comp):
    PCA_emp = comp[i]
    for j in range(n_comp_sim):
        PCA_sim = comp_sim[j]
        correlation_matrix[i][j] = np.absolute(pearsonr(PCA_emp/np.sum(PCA_emp), PCA_sim/np.sum(PCA_sim))[0])
# Plot matrix
plot_PC_correlation_matrix(correlation_matrix, n_comp, n_comp_sim,
        save_path=f"{save_analysis}/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_PCA_emp_sim_corr_matrix_envelope.png")
# Extract the PC_emp, PC_sim pair with the highest correlation and plot
i, j = np.unravel_index(np.argmax(correlation_matrix, axis=None), correlation_matrix.shape)
max_corr_val_slp = np.max(correlation_matrix)
PCA_emp_slp = comp[i]
PCA_sim_slp = comp_sim[j]
plot_PCA_emp_sim(i, j, max_corr_val_slp, PCA_emp_slp, PCA_sim_slp, ch_names_common, comp_variance, comp_sim_variance,
     save_path=f"{save_analysis}/Fig3_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_PCA_emp_sim_envelope.png")
print('Correlation envelopes PCA1emp and PCA1sim = ', max_corr_val_slp)#calc_correlation(PCA_emp, PCA_sim))
# Correlation between two first PCA components
PCA1_corr_val_slp = correlation_matrix[0][0]

#%%  Binarize SEEG timeseries into a 2D binary image, where 0: no seizure, 1: seizure
# we only take into account the window from seizure onset until seizure offset (so we remove the base length)
plot_binary = False
start = int(base_length * seeg_info['sfreq'])
end = slp_c.shape[0] - int(base_length * seeg_info['sfreq'])
binarized2Dslp = np.zeros(shape=slp_c[start:end, :].shape, dtype=int)
for ch_idx in range(len(ch_names_common)):
    if ez_channels_slp_c[ch_idx] == 1:
        binarized2Dslp[onsets_c[ch_idx]-start:offsets_c[ch_idx]-start, ch_idx] = 1 # TODO run analysis again on this new method
# plot_binarized_SEEG(t, ch_names_common, onsets_c, offsets_c, ez_channels_slp_c, seeg_info,
        # save_path=f"{save_analysis}/Fig4_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_binarizedSEEG_emp.png")
if plot_binary:
    plot_2d_binary_SEEG(binarized2Dslp, ch_names_common, scaleplt=0.1,
        save_path=f"{save_analysis}/Fig4_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_binarizedSEEG_emp.png")

# Binarize simulated SEEG timeseries into a 2D binary image, where 0: no seizure, 1: seizure
# we only take into account the first start of the first seizure channel until the last seizure offset
t_min_seizure_start_sim = min([t_sim[onsets_sim_c[i]] for i in np.where(ez_channels_slp_sim_c == 1)[0]])
start_sim = int(t_min_seizure_start_sim * sfreq_sim)
t_max_seizure_end_sim = max([t_sim[offsets_sim_c[i]] for i in np.where(ez_channels_slp_sim_c == 1)[0]])
end_sim = int(t_max_seizure_end_sim * sfreq_sim)
binarized2Dslp_sim = np.zeros(shape=slp_sim_c[start_sim:end_sim, :].shape, dtype=int)
for ch_idx in range(len(ch_names_common)):
    if ez_channels_slp_sim_c[ch_idx] == 1:
        binarized2Dslp_sim[onsets_sim_c[ch_idx]-start_sim:offsets_sim_c[ch_idx]-start_sim, ch_idx] = 1 # TODO run analysis again on this new method
# plot_binarized_SEEG(t_sim, ch_names_common, onsets_sim_c, offsets_sim_c, ez_channels_slp_sim_c,
#         save_path=f"{save_analysis}/Fig4_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_binarizedSEEG_sim.png")
if plot_binary:
    plot_2d_binary_SEEG(binarized2Dslp_sim, ch_names_common, scaleplt=0.1,
        save_path=f"{save_analysis}/Fig4_{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_binarizedSEEG_sim.png")

# Upsampling simulated data to empirical data
sf = sfreq_sim
# datetime = pd.date_range(start='00:00:00', end=f'00:00:{t_sim[-1]}', periods=t_sim.size)
datetime = pd.date_range(start='00:00:00', end=f'00:00:{t_sim[end_sim]}', periods=t_sim[start_sim:end_sim].size)
df = pd.DataFrame({'Datetime': datetime})
for i in range(binarized2Dslp_sim.shape[1]):
    df[f'value{i}'] = binarized2Dslp_sim[:, i]
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace = True)
sf_new = t[start:end].shape[0]/t_sim[start_sim:end_sim].shape[0]
df_new = df.resample(f'{np.round(1/sf_new,6)}ms').mean().reset_index()
df_interpolated = df_new.interpolate(method='linear')
binarized2Dslp_sim_new = np.asarray(df_interpolated[df_interpolated.columns[1:binarized2Dslp_sim.shape[1]+1]],
                                    dtype='int')
# plot_2d_binary_SEEG(binarized2Dslp_sim_new, ch_names_common, scaleplt=0.1)

# STATS
if binarized2Dslp_sim_new.shape[0] <= binarized2Dslp.shape[0]:
    blur = binarized2Dslp_sim_new
    org = binarized2Dslp[:binarized2Dslp_sim_new.shape[0],:]
else:
    blur = binarized2Dslp_sim_new[:binarized2Dslp.shape[0],:]
    org = binarized2Dslp
agreement_2D = np.sum(blur == org)/org.size
correlation_2D = np.corrcoef(blur.flat, org.flat)[0, 1]
mse_2D = mse(blur,org)
rmse_2D = rmse(blur, org)
print('Agreement', agreement_2D)
print('Correlation on flattened images', correlation_2D)
print("MSE: ", mse_2D)
print("RMSE: ", rmse_2D)

#%% SAVE STATS IN DATAFRAME
option = 1 # TODO change if needed
row_index = 23

# df = pd.DataFrame(columns=['subject_id', 'emp_seizure', 'sim_seizure',
#                            'corr_envelope_amp', 'corr_signal_pow', 'binary_overlap',
#                            'SO_overlap', 'SP_overlap', 'NS_overlap', 'PCA_correlation'])
filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim.csv')
df = pd.read_csv(filepath)

## Add new column
# df.insert(19, "2D_rmse", np.zeros(24))

if option == 1: # Here we append new row at the end of the dataframe
    new_row = {'subject_id': pid_bids, 'emp_seizure': szr_names[szr_index].strip("' "),
               'sim_seizure': f'{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg',
               'corr_envelope_amp': corr_envelope_amp, 'corr_signal_pow': corr_signal_pow, 'binary_overlap': overlap,
               'SO_overlap': occurrence_SO, 'SP_overlap': occurrence_SP, 'NS_overlap': occurrence_NS,
               'PCA_correlation': max_corr_val, 'PCA1_correlation': PCA1_corr_val,
               'PCA_correlation_slp': max_corr_val_slp, 'PCA1_correlation_slp': PCA1_corr_val_slp,
               'J_SO_overlap': jaccard_coeff_SO, 'J_SP_overlap': jaccard_coeff_SP, 'J_NS_overlap': jaccard_coeff_NS,
               '2D_agreement': agreement_2D, '2D_correlation': correlation_2D, '2D_mse': mse_2D, '2D_rmse': rmse_2D}
    df2 = df.append(new_row, ignore_index=True)

    # save dataframe
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim.csv')
    df2.to_csv(filepath, index=False)

elif option == 2: # Here we change values for a specific row
# TODO NOTE FOR FUTRE UPDATES UPDATE ROW INDEX !!!
# To update the values of the rows from index 0 to 2 with respect to columns Num and NAME respectively
# data.loc[0:2,['Num','NAME']] = [100,'Python']

    df.loc[row_index, ['corr_envelope_amp', 'corr_signal_pow', 'binary_overlap', 'SO_overlap', 'SP_overlap',
                       'NS_overlap', 'PCA_correlation', 'PCA1_correlation', 'PCA_correlation_slp', 'PCA1_correlation_slp',
                       'J_SO_overlap', 'J_SP_overlap', 'J_NS_overlap', '2D_agreement', '2D_correlation', '2D_mse',
                       '2D_rmse']] = [corr_envelope_amp, corr_signal_pow, overlap, occurrence_SO, occurrence_SP,
                                      occurrence_NS, max_corr_val, PCA1_corr_val, max_corr_val_slp, PCA1_corr_val_slp,
                                      jaccard_coeff_SO, jaccard_coeff_SP, jaccard_coeff_NS, agreement_2D, correlation_2D,
                                      mse_2D, rmse_2D]
    # save dataframe
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim.csv')
    df.to_csv(filepath, index=False)


################### FINALS
# plot stats accross patients
hypothesis = 'VEPhypothesis'
# hypothesis = 'clinicalhypothesis'

#%% VEC DATASET
filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim.csv')
df = pd.read_csv(filepath)
binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list,\
agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = compute_stats(df, hyp=hypothesis, plot=False)
# plot_avg_stats(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, corr_env_list,
#                       corr_signal_pow_list, PCA_correlation_list, agreement_img_list, correlation_img_list, hyp=hypothesis)

# plot_avg_stats_overlap(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list,
#            J_SP_overlap_list, J_NS_overlap_list, hyp=hypothesis)
#
# plot_avg_stats_correlation(corr_env_list, corr_signal_pow_list, PCA_correlation_list, PCA1_correlation_list,
#             PCA_correlation_slp_list, PCA1_correlation_slp_list, hyp=hypothesis)
#
# plot_avg_stats_binary_img(agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, hyp=hypothesis)

#%% CONTROL DATASET
filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_control.csv')
df_control = pd.read_csv(filepath)
binary_overlap_listc, SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, J_SO_overlap_listc, \
J_SP_overlap_listc, J_NS_overlap_listc, corr_env_listc, corr_signal_pow_listc, \
PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc,\
agreement_img_listc, correlation_img_listc, mse_img_listc, rmse_img_listc = compute_stats(df_control, hyp=hypothesis, plot=False)


plot_avg_stats_overlap_two_datasets(binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list,
                                        J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, binary_overlap_listc,
                                        SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, J_SO_overlap_listc,
                                        J_SP_overlap_listc, J_NS_overlap_listc, hyp=hypothesis)

plot_avg_stats_correlation_two_datasets(corr_env_list, corr_signal_pow_list, PCA_correlation_list,
    PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, corr_env_listc, corr_signal_pow_listc,
    PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc, hyp=hypothesis)

plot_avg_stats_binary_img_two_datasets(agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list,
    agreement_img_listc, correlation_img_listc, mse_img_listc, rmse_img_listc, hyp=hypothesis)



# FOR PAPER FIGURE ERASEME LATER

# TODO complete with all data points
fig, ax = plt.subplots()
plt.title(f'{hypothesis}', fontsize=18)
trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
ax.errorbar([0, 1, 2, 3], [np.mean(correlation_img_list), np.mean(binary_overlap_list), np.mean(J_SO_overlap_list)/100, np.mean(J_SP_overlap_list)/100],
            [np.std(correlation_img_list), np.std(binary_overlap_list), np.std(J_SO_overlap_list)/100,
             np.std(J_SP_overlap_list)/100], fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0,
            transform=trans1, label='VEC')
ax.scatter(np.zeros(shape=len(correlation_img_list)), correlation_img_list, transform=trans1, color='lightblue')
ax.scatter(np.ones(shape=len(binary_overlap_list)), binary_overlap_list, transform=trans1, color='lightblue')
ax.scatter(np.ones(shape=len(J_SO_overlap_list))*2, np.asarray(J_SO_overlap_list)/100, transform=trans1, color='lightblue')
ax.scatter(np.ones(shape=len(J_SP_overlap_list))*3, np.asarray(J_SP_overlap_list)/100, transform=trans1, color='lightblue')

ax.errorbar([0, 1, 2, 3],
            [np.mean(correlation_img_listc), np.mean(binary_overlap_listc), np.mean(J_SO_overlap_listc)/100, np.mean(J_SP_overlap_listc)/100],
            [np.std(correlation_img_listc), np.std(binary_overlap_listc), np.std(J_SO_overlap_listc)/100,
             np.std(J_SP_overlap_listc)/100], fmt='o', ecolor='tomato', color='black', elinewidth=3, capsize=0,
            transform=trans2, label='Control')
ax.scatter(np.zeros(shape=len(correlation_img_listc)), correlation_img_listc, transform=trans2, color='mistyrose')
ax.scatter(np.ones(shape=len(binary_overlap_listc)), binary_overlap_listc, transform=trans2, color='mistyrose')
ax.scatter(np.ones(shape=len(J_SO_overlap_listc))*2, np.asarray(J_SO_overlap_listc)/100, transform=trans2, color='mistyrose')
ax.scatter(np.ones(shape=len(J_SP_overlap_listc))*3, np.asarray(J_SP_overlap_listc)/100, transform=trans2, color='mistyrose')

plt.xticks([0, 1, 2, 3], ['Correlation', 'Overlap', 'J_SO', 'J_SP'], fontsize=18)
plt.ylabel('Emp/Sim Correlation', fontsize=18)
plt.ylim([0, 1])
plt.tight_layout()
plt.legend(fontsize=18)
plt.show()

# P-value calculation
# Calculates the T-test for the means of two independent samples of scores.
# This is a two-sided test for the null hypothesis that two independent samples have identical average (expected) values.
# This test assumes that the populations have identical variances by default.
print(stats.ttest_ind(correlation_img_list,correlation_img_listc))
print(stats.ttest_ind(binary_overlap_list,binary_overlap_listc))
print(stats.ttest_ind(J_SO_overlap_list,J_SO_overlap_listc))
print(stats.ttest_ind(J_SP_overlap_list,J_SP_overlap_listc))

