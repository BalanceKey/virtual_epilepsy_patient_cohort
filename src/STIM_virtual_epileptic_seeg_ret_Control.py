'''
Generate control data, by taking the brain structure of one patient, and generating simulations with
random EZ hypothesis from other patients. Then map this onto SEEG of this patient and compute metrics again.
'''

#!/usr/bin/env python
# Importing libraries
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.image as mpimg
import colorednoise as cn
import os.path as op
from pybv import write_brainvision
import pandas as pd
import time as tm
import sys
import csv
import json
from src.utils import isfloat
from src.utils_functions import vep_prepare_ret
from src.utils_functions.model import EpileptorStim
from src.utils_functions.model_2populations import EpileptorStim2Populations
from src.utils_functions.integrator import HeunDeterministicAdapted, HeunStochasticAdapted
roi = vep_prepare_ret.read_vep_mrtrix_lut()

clinical_hypothesis = False
run = 5

# # 1. Importing patient specific information
# - Choosing a patient to work on.
# - Defining the folder to save the results for the patient.
# - Importing the structural connectivity, the list of regions from the parcellation.

pid = 'id003_mg'  # 'id005_ft'#'id003_mg'
pid_bids = f'sub-{pid[2:5]}'
subj_proc_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'
save_data = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/ControlCohort/'
os.makedirs(save_data, exist_ok=True)

try:
    if op.exists(f'{subj_proc_dir}/dwi_new/weights.txt'):
        print("Careful here taking connectivity from diw_new")
    else:
        print("taking connectivity from tvb")
except FileNotFoundError as err:
    print(f'{err}: Structural connectivity not found for {subj_proc_dir}')

# Connectivity
con = connectivity.Connectivity.from_file(str(subj_proc_dir + '/tvb/connectivity.vep.zip'))
con.tract_lengths = np.zeros((con.tract_lengths.shape))  # no time-delays
con.weights[np.diag_indices(con.weights.shape[0])] = 0
# con.weights = np.log(con.weights+1)
con.weights /= con.weights.max()
con.configure()

fig = plt.figure(figsize=(20, 20))
plt.imshow(con.weights, norm=plc.LogNorm(vmin=1e-6, vmax=con.weights.max()))
plt.title(f'{pid}: Normalized SC (log scale)', fontsize=12, fontweight='bold')
plt.xticks(np.r_[:len(roi)], roi, rotation=90)
plt.yticks(np.r_[:len(roi)], roi)
fig.tight_layout()
save_sc = False
if save_sc:
    print('>> Save', f'{save_struct}/img/{pid_bids}_SC_matrix.png')
    plt.savefig(f'{save_struct}/img/{pid_bids}_SC_matrix.png')
    np.savez(f'{save_struct}/{pid_bids}_SC_matrix', SC=con.weights, roi=roi)
plt.show()

fig = plt.figure(figsize=(25, 10))
reg_name = 'Right-Hippocampus-anterior'
img = plt.bar(np.r_[0:con.weights.shape[0]], con.weights[roi.index(reg_name)], color='black', alpha=0.3)
plt.title(f'Connection weights to {reg_name}', fontsize=40)
plt.xticks(np.r_[:len(roi)], roi, rotation=90)
plt.xlabel('Region#', fontsize=30)
fig.tight_layout()
plt.show()

# # 2. Choose a seizure to model
szr_name = f'{subj_proc_dir}/seeg/fif/170614C-CEX_0017_1.json'
# Load seizure data
seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subj_proc_dir, szr_name)

# plot real data at sensor level with 5 sec before and after
base_duration = 5
scaleplt = 0.001
f = vep_prepare.plot_sensor_data(bip, gain_prior, seeg_info, base_duration, data_scaleplt=scaleplt,
                                 title=f'{pid}:ts_real_data')
plt.show()

plt.imshow(mpimg.imread(f'{subj_proc_dir}/elec/elec.vep.png'))
plt.show()

nsensor, nchan = np.shape(gain)
fig = plt.figure(figsize=(20, 20))
im = plt.imshow(gain, norm=plc.LogNorm(vmin=gain.min(), vmax=gain.max()))
plt.xticks(np.r_[:len(roi)], roi, rotation=90)
plt.yticks(np.r_[:len(bip.ch_names)], bip.ch_names)
plt.xlabel('Region#', fontsize=12)
plt.ylabel('Channel#', fontsize=12)
plt.title(f'{pid_bids}: Gain Matrix (log scale)', fontsize=12, fontweight='bold');
plt.colorbar(im, fraction=0.03, pad=0.01)
fig.tight_layout()
save_gain = False
if save_gain:
    print('>> Save', f'{save_struct}/img/{pid_bids}_gain.png')
    plt.savefig(f'{save_struct}/img/{pid_bids}_gain.png')
plt.show()

# # 3. Setting up the simulation with TVB
# - Here we use the Epileptor
# - We set up the needed ingredients for the simulator (model, connectivity, coupling, integrator, initial conditions, monitors and more)
ses = "ses-02"
task = "simulatedstimulation"
if clinical_hypothesis:
    sub_folder = 'clinical_hypothesis'
    acq = "clinicalhypothesis"
    print()
else:
    sub_folder = 'vep_hypothesis'
    acq = "VEPhypothesis"

print(f'Simulation using EZ from \"{sub_folder}\".')

# ## 3b. Configure the stimulation
# - Select a pair of electrodes to apply the stimulation from
# - Calculate the effect of those electrodes' stimulation on the network nodes using the gain_prior matrix

# TODO We take parameters from the patient's saved simulation parameters
# TODO We take the run 01 simulation by default
VEC_patient_data = f'{save_data}/../VirtualEpilepticCohort/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/'
simulator_parameters_path = f'{VEC_patient_data}/{pid_bids}_simulator_parameters_run-01.tsv'
epileptor_parameters_path = f'{VEC_patient_data}/{pid_bids}_epileptor_parameters_run-01.tsv'
stimulation_parameters_path = f'{VEC_patient_data}/{pid_bids}_stimulation_parameters_run-01.tsv'
simulator_parameters = pd.read_csv(simulator_parameters_path, sep='\t')
epileptor_parameters = pd.read_csv(epileptor_parameters_path, sep='\t')
stimulation_parameters = pd.read_csv(stimulation_parameters_path, sep='\t')

# slopestring = [el.strip('][') for el in epileptor_parameters['slope'][0].split(' ')]
# slopearr = np.asarray([float(char) for char in slopestring if isfloat(char)])

initcondstring = [el.strip('][') for el in simulator_parameters['init_cond'][0].split(' ')]
initcondarr = np.asarray([float(char) for char in initcondstring if isfloat(char)])

epileptor_x0string = [el.strip('][') for el in epileptor_parameters['x0'][0].split(' ')]
epileptor_x0arr =  np.asarray([float(char) for char in epileptor_x0string if isfloat(char)])
assert epileptor_x0arr.size == 162

stimulation_weights_string = [el.strip('[]') for el in stimulation_parameters['stimulation_weights'][0].split(' ')]
stimulation_weightsarr = np.asarray([float(char) for char in stimulation_weights_string if isfloat(char)])
assert stimulation_weightsarr.size == 162 # making sure there are as many weights as regions

noise_coeffs_string = [el.strip('][') for el in simulator_parameters['noise_coeffs'][0].split(' ')]
noise_coeffs_arr = np.asarray([float(char) for char in noise_coeffs_string if isfloat(char)])


# Calculate the projection of the SEEG electrode stimulation onto the network nodes
# choi = "B'1-2"  # "C'2-3"
# idx = bip.ch_names.index(choi)
# fig = plt.figure(figsize=(25, 10))
# img = plt.bar(np.r_[0:gain_prior[idx].shape[0]], gain_prior[idx], color='blue', alpha=0.5)
# plt.xticks(np.r_[:len(roi)], roi, rotation=90)
# plt.xlabel('Region#', fontsize=30);
# plt.ylabel('Gain matrix for channel ' + choi, fontsize=30);
# fig.tight_layout()
# plt.show()
#
# maxval = gain_prior[idx].max()
# maxval_roi = roi[np.where(gain_prior[idx] == maxval)[0][0]]
# print('The maximum impact of the forward solution is on: ' + maxval_roi + ':',
#       gain_prior[idx][np.where(roi == 'Left-Hippocampus-anterior')])
# print('Impact on Left-Hippocampus-posterior : ', gain_prior[idx][roi.index('Right-Hippocampus-posterior')])

# Stimuli parameters
# freq = 50 # Hz
# sfreq = 400 # how many steps there are in one second
dt = float(simulator_parameters['dt'])#0.05
onset = float(stimulation_parameters['onset'])#2 * sfreq
# n_sec = 5 # stim duration
stim_length = float(stimulation_parameters['stimulation_length']) #n_sec * sfreq + onset # n_sec * sfreq + onset
simulation_length = float(simulator_parameters['simulation_length'])#50 * sfreq # n_sec * sfreq
T = float(stimulation_parameters['T']) # 1/freq * sfreq # pulse repetition period [ms]
tau = float(stimulation_parameters['pulse_width'])#2  # pulse duration [ms]
I =  float(stimulation_parameters['I']) #2  # pulse intensity [mA]

# stim_reg_id = np.where(roi == 'Right-Hippocampus-anterior')[0]
n_regions = len(con.region_labels)
class vector1D(equations.DiscreteEquation):
    equation = equations.Final(default="emp")
# stimulus parameters
eqn_t = vector1D()
parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset}
pulse1, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
pulse1_ts = [p[1] for p in pulse1]
parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset + tau}
pulse2, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
pulse2_ts = [p[1] for p in pulse2]
pulse_ts = np.asarray(pulse1_ts) - np.asarray(pulse2_ts)
stimulus_ts = np.hstack((pulse_ts[:-1], np.zeros(int(np.ceil((simulation_length - stim_length) / dt)))))
eqn_t.parameters['emp'] = np.copy(stimulus_ts)

# Stimuli applied from the SEEG electrode
# print("Stimuli applied from the SEEG electrode")
# plt.figure(figsize=(30, 1))
# plt.plot(stimulus_ts)
# plt.figure(figsize=(30, 1))
# plt.plot(stimulus_ts[0:20000])
# plt.show()

# Stimuli weighted accross the network nodes according to the gain matrix
stim_weights = stimulation_weightsarr#(gain_prior[idx] - gain_prior[idx].min()) / (gain_prior[idx].max() - gain_prior[idx].min())
stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                  connectivity=con,
                                  weight=stim_weights)
stimulus.configure_space()
stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))
# plt.show()

print("Stimuli weighted accross the network nodes according to the gain matrix")
def plot_pattern(pattern_object):
    """
    pyplot in 2D the given X, over T.
    """
    plt.figure(figsize=(25, 45), tight_layout=True)
    plt.subplot(411)
    #     plt.plot(pattern_object.spatial_pattern, "k*")
    plt.bar(np.r_[0:pattern_object.spatial_pattern.shape[0]], pattern_object.spatial_pattern[:, 0])
    plt.xticks(np.r_[:len(roi)], roi, rotation=90, size=12)
    plt.title("Space", fontsize=26)
    #     plt.plot(pattern_object.space, pattern_object.spatial_pattern, "k*")
    plt.subplot(412)
    plt.plot(pattern_object.time.T, pattern_object.temporal_pattern.T, linewidth=1)
    plt.title("Time", fontsize=26)
    # plt.show()

    # plt.figure(figsize=(30, 25))
    plt.subplot(212)
    plt.imshow(pattern_object(), aspect="auto", cmap='nipy_spectral')
    # plt.colorbar()
    plt.title("Stimulus", fontsize=26)
    plt.yticks(np.r_[:len(roi)], roi)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Space", fontsize=16)
    plt.show()
plot_pattern(stimulus)

# epileptors = Epileptor3D4(variables_of_interest=['x1', 'y1', 'z', 'm'])
epileptors = EpileptorStim2Populations(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'm'])
seizure_length = seeg_info['offset'] - seeg_info['onset']
print(f"Seizure is {seizure_length} s long")
epileptors.r = np.array([epileptor_parameters['r'][0].strip('[]')], dtype=float)#np.array([0.000025])#np.array([0.00002])
r_val = float(epileptor_parameters['r2'][0].split(' ')[0].strip('[]'))#0.0004
epileptors.r2 = np.ones(len(roi)) * (r_val)
epileptors.Istim = np.ones(len(roi)) * (0.)
epileptors.n_stim = np.zeros(len(roi))
epileptors.Iext = np.ones(n_regions) * (3.1)
epileptors.Iext2 = np.ones(n_regions) * (0.45)
epileptors.Ks = np.ones(n_regions)
epileptors.Kf = np.ones(n_regions) * 0.001  # ?
epileptors.Kvf = np.ones(n_regions) * 0.01  # ?
epileptors.threshold = np.ones(len(roi)) * (10.)

period = float(simulator_parameters['period'])#1.
cpl_factor = float(simulator_parameters['coupling_factor'])#-0.01 # -4
# dt = 0.05 # already defined above
x0ez = -2.0    #-1.3
x0pz = -2.0  #-1.65
x0pz2 = -2.0  #-1.83
x0num = -2.2
epileptors.x0 = np.ones(n_regions) * x0num

# We first check the EZ hypothesis used for the VEC patient, and then compare to random control
# To make sure they aren't identical
if clinical_hypothesis:
    print('Clinical Hypothesis')
    ezfile = json.load(open(f'{subj_proc_dir}/../ei-vep_53.json'))
    ez = ezfile[pid]['ez']
    print('EZ: ', ez)

else:
    print('VEP Hypothesis')
    ezfile = pd.ExcelFile(f'{subj_proc_dir}/../vep_pipeline_heatmap.xlsx', engine='openpyxl')
    ezfile = pd.read_excel(ezfile, pid)
    ezroi, ezval = np.array(ezfile['Brain Regions']), np.array(ezfile['Evs'])
    ez = [ezroi[iez] for iez in np.where(ezval >= 0.5)[0]]
    pz = [ezroi[iez] for iez in np.where((ezval >= 0.2) & (ezval < 0.5))[0]]
    pz2 = [ezroi[iez] for iez in np.where((ezval > 0.05) & (ezval < 0.2))[0]]
    print('EZ: ', ez)
    print('PZ: ', pz)
    print('PZ2: ', pz2)

# TODO NOTE Here we take random ez hypothesis from other patient data
samples = 23
rand_patient_id = np.random.choice(samples, size=1, replace=False)
rand_pid_bids = 'sub-'+'{:0>3}'.format(rand_patient_id[0]+1)
rand_patient_data = f'{save_data}/../VirtualEpilepticCohort/derivatives/tvb/{rand_pid_bids}/ses-01/{acq}/parameters/'
rand_epileptor_param_path = f'{rand_patient_data}/{rand_pid_bids}_epileptor_parameters_run-01.tsv'
rand_epileptor_parameters = pd.read_csv(rand_epileptor_param_path, sep='\t')
# TODO check this rand patient is not a patient we already simulated
assert rand_pid_bids != pid_bids
ez_rand = [el.strip("['] ") for el in rand_epileptor_parameters['EZ'][0].split(',')]
pz_rand = [el.strip("['] ") for el in rand_epileptor_parameters['PZ'][0].split(',')]
if pz_rand == ['']:
    pz_rand = []
print(rand_pid_bids)
print(ez_rand)
print(pz_rand)

# Getting the index for the ez & pz regions from the RANDOM patient's parameters
index_ez = [roi.index(iez) for iez in ez_rand]
index_pz = [roi.index(iez) for iez in pz_rand]
# index_pz2 = [roi.index(iez) for iez in pz2]

# Defining the excitability for those regions
epileptors.x0[index_ez] = x0ez
epileptors.x0[index_pz] = x0ez
# epileptors.x0[index_pz2] = x0pz2

# TODO remove this later
# epileptors.x0[roi.index('Left-Frontal-pole')] = -3.
# epileptors.x0[roi.index('Left-Lingual-gyrus')] = -3.
# epileptors.x0[roi.index('Left-Cerebellar-cortex')] = -3.
# epileptors.x0[roi.index('Right-Cerebellar-cortex')] = -3.
# epileptors.x0[roi.index('Right-Fusiform-gyrus')] = -3.
# epileptors.x0[roi.index('Right-Lingual-gyrus')] = -3.
# epileptors.x0[roi.index('Left-Occipital-pole')] = -3.

epileptors.threshold[index_ez] = 1.5
epileptors.threshold[index_pz] = 1.8
# epileptors.threshold[index_pz2] = 2.5

# epileptors.n_stim[index_ez] = 1
epileptors.n_stim = np.ones(len(roi)) # TODO fix this once and for all by removing it !!!

# Coupling
coupl = coupling.Difference(a=np.array([cpl_factor]))

# Integrator
noisy_sim = True
if noisy_sim:
    #     hiss = noise.Additive(nsig = np.array([0.02, 0.02, 0., 0.0003, 0.0003, 0.]))
    #     nsig = np.array([0.005, 0.005, 0., 0.00005, 0.00005, 0.]) # little to no interictal spikes
    nsig = noise_coeffs_arr# np.array([0.005, 0.005, 0., 0.0001, 0.0001, 0., 0.])
    hiss = noise.Additive(nsig=nsig)
    hiss.ntau = 1  # for gaussian distributed coloured noise
    heunint = HeunStochasticAdapted(dt=dt, noise=hiss)
else:
    nsig = []
    heunint = HeunDeterministicAdapted(dt=dt)

# Monitors
mon_tavg = monitors.TemporalAverage(period=period)

# ic = [-1.46242601e+00,  -9.69344913e+00,   2.98, 0]
ic = [-1.46242601e+00, -9.69344913e+00, 2.97, -1.05214059e+00, -4.95543740e-20, -1.98742113e-01, 0.0]
ic_full = np.repeat(ic, len(roi)).reshape((1, len(ic), len(roi), 1))
# ic_full[0,:,roil.index('Left-T3-posterior'),0] = [-1.46242601e+00,  -9.69344913e+00,   2.98, -0.2]

# Simulator
sim = simulator.Simulator(model=epileptors,
                          initial_conditions=ic_full,
                          connectivity=con,
                          stimulus=stimulus,
                          coupling=coupl,
                          integrator=heunint,
                          conduction_speed=np.inf,
                          monitors=[mon_tavg])

sim.configure()

print("Starting simulation...")
tic = tm.time()
ttavg = sim.run(simulation_length=simulation_length)
print("Finished simulation.")
print('execute for ', (tm.time() - tic) / 60.0, 'mins')
save_fig = False

# # 4. Visualizing the simulated data on the source level
tts = ttavg[0][0]
tavg = ttavg[0][1]
srcSig = tavg[:, 0, :, 0] - tavg[:, 3, :, 0]
start_idx = 0
end_idx = int(simulation_length)
srcSig_normal = srcSig / np.ptp(srcSig)
# Plot raw time series
figure = plt.figure(figsize=(20, 40))
plt.plot(tts[start_idx:end_idx], srcSig_normal[start_idx:end_idx] + np.r_[:162], 'r')
plt.title("Epileptors time series")
plt.yticks(np.arange(len(roi)), roi, fontsize=22)
plt.xticks(fontsize=22)
plt.ylim([-1, len(roi) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:{sub_folder}')
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_source_timeseries_run-0{run}.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_source_timeseries_run-0{run}.png')
plt.show()

# idx = np.where(roi == 'Left-Hippocampus-anterior')[0][0]
# idx = np.where(roi == 'Right-Insula-gyri-longi')[0][0]
# idx = np.where(roi == 'Left-O2')[0][0]
# id = roi.index('Right-Superior-parietal-lobule-P1')
# id = roi.index('Left-Hippocampus-anterior')
id = roi.index('Right-Rhinal-cortex')
# id = roi.index('Left-STS-anterior')
# id = roi.index('Left-Insula-gyri-brevi')
print(id)
y = tavg[:, :, [id], 0]
print(tavg.shape)
print('Region ' + roi[id])
fig, axs = plt.subplots(4, 1, tight_layout=True, figsize=(9, 10))
axs[0].set_title(f"{roi[id]} I={I}, T={T}, tau={tau}, r={r_val}, ", fontsize=15)
axs[0].plot(tts[:], y[:, 0, :] - y[:, 3, :], 'C3', linewidth=0.5)
axs[0].set_ylabel('x1-x2')
axs[1].plot(tts[:], y[:, 2, :], 'C2', linewidth=1.5)
axs[1].set_ylabel('z')
axs[2].plot(tts[:], y[:, 6, :], 'C4', linewidth=1.5)
axs[2].axhline(y=2.5, linewidth=1.5)
axs[2].set_ylabel('m')
axs[3].plot(stimulus.time.T, stimulus.temporal_pattern.T * stimulus.weight[id], 'C5', linewidth=0.5)
axs[3].set_ylabel('I_stim')
# axs[5].plot(t[:], np.ones(simulation_length)*-2.2, 'C6', linewidth=0.5)
# axs[5].set_ylabel('x0')
for i in range(4):
    axs[i].set_xlabel('Time [ms]', fontsize=7)
plt.show()


# # 5. Visualizing the data on the sensor level
seeg = np.dot(gain, srcSig.T)
if clinical_hypothesis:
    basicfilename = 'ClinicalHypothesis'
else:
    basicfilename = 'VEP_Hypothesis'
show_ch = bip.ch_names
nch = [show_ch.index(ichan) for ichan in show_ch]
nch_sourse = []
for ind_ch, ichan in enumerate(show_ch):
    isource = roi[np.argmax(gain_prior[ind_ch])]
    nch_sourse.append(f'{isource}:{ichan}')

plt.figure(figsize=[40, 70])
scaleplt = 0.04
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (seeg[ich, start_idx:end_idx] - seeg[ich, 0]) + ind, 'blue', lw=0.9);
plt.xticks(fontsize=18)
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26);
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
plt.show()

# Applying a high pass filter to the signal to resseble AC recordings
import scipy.signal as signal
def highpass_filter(y, sr):
    """In this case, the filter_stop_freq is that frequency below which the filter MUST act like a stop filter and filter_pass_freq is that frequency above which the filter MUST act like a pass filter.
       The frequencies between filter_stop_freq and filter_pass_freq are the transition region or band."""
    filter_stop_freq = 2  # Hz
    filter_pass_freq = 2  # Hz
    filter_order = 1001

    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

    # Apply high-pass filter
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

plt.figure(figsize=[40, 70])
scaleplt = 0.15
# start_idx = 2100
# end_idx = 7500#len(seeg)
y = highpass_filter(seeg, 48)  # seeg
beta = 1  # the exponent
noise1 = cn.powerlaw_psd_gaussian(beta, y.shape)
beta = 2  # the exponent
noise2 = cn.powerlaw_psd_gaussian(beta, y.shape)
beta = 4  # the exponent
noise3 = cn.powerlaw_psd_gaussian(beta, y.shape)
# y_new = y_filt + noise + noise2
y_new = y# + noise1 * 0.1 + noise2 * 0.1
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (y_new[ich, start_idx:end_idx] - y_new[ich, 0]) + ind, 'blue', lw=1.5)
# plt.xticks(fontsize=18)
plt.xticks([], [])
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
TS_on = start_idx + 2500
TS_off = end_idx - 5000
# plt.axvline(TS_on, color='DeepPink', lw=2)
# plt.axvline(TS_off, color='DeepPink', lw=2)
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
plt.show()

# # 6. Saving the data in BIDS format
# Do this once you are happy with what you got.
#
# Once all the data has been generated, for one patient, it's time to convert it to BIDS format
#
# To check:
#
# bids-validator BIDS/VirtualCohortProject --config.ignore=99

print(ses + ' ' + task + ' ' + acq + ' run', run)

os.makedirs(save_data, exist_ok=True)
os.makedirs(f'{save_data}/{pid_bids}', exist_ok=True)
os.makedirs(f'{save_data}/{pid_bids}/{ses}', exist_ok=True)
os.makedirs(f'{save_data}/{pid_bids}/{ses}/ieeg', exist_ok=True)

# saving other data into derivatives
os.makedirs(f'{save_data}/derivatives', exist_ok=True)
os.makedirs(f'{save_data}/derivatives/tvb', exist_ok=True)
os.makedirs(f'{save_data}/derivatives/tvb/{pid_bids}', exist_ok=True)
save_struct = f'{save_data}/derivatives/tvb/{pid_bids}/struct'
os.makedirs(save_struct, exist_ok=True)  # to save SC and gain matrix
os.makedirs(f'{save_struct}/img/', exist_ok=True)  # to save SC and gain matrix images
os.makedirs(f'{save_data}/derivatives/tvb/{pid_bids}/{ses}',
            exist_ok=True)  # to save seizure data, one simulation type (seizure, stimulation etc.) per session
os.makedirs(f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}', exist_ok=True)
save_img = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/img'
os.makedirs(save_img, exist_ok=True)
os.makedirs(f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters', exist_ok=True)

print(start_idx)
print(end_idx)

epi_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_epileptor_parameters_run-0{run}'
sim_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_simulator_parameters_run-0{run}'
stim_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_stimulation_parameters_run-0{run}'

# model and simulator parameters to run with TVB
print('Saving: ' + epi_params_file)
with open(f'{epi_params_file}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['rand_patient_id', 'rand_patient_name', 'EZ', 'PZ', 'x0', 'Iext', 'Iext2', 'r', 'r2', 'Ks', 'Kf', 'Kvf'])
    tsv_writer.writerow([rand_patient_id[0], rand_pid_bids, ez_rand, pz_rand, epileptors.x0, epileptors.Iext, epileptors.Iext2, epileptors.r, epileptors.r2,
                         epileptors.Ks, epileptors.Kf, epileptors.Kvf])

print('Saving: ' + sim_params_file)
with open(f'{sim_params_file}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['coupling_factor', 'noise_coeffs', 'init_cond', 'dt', 'period', 'simulation_length'])
    tsv_writer.writerow([cpl_factor, nsig, ic_full, dt, period, simulation_length])

print('Saving: ' + stim_params_file)
with open(f'{stim_params_file}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['stimulation_weights', 'stimulation_length', 'onset', 'T', 'I', 'pulse_width'])
    tsv_writer.writerow([stim_weights, stim_length, onset, T, I, tau])

# save the generated data on the source level
save_src_timeseries = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/{pid_bids}_simulated_source_timeseries_run-0{run}'

print('Saving: ' + save_src_timeseries)
with open(f'{save_src_timeseries}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['time_steps', 'source_signal'])
    tsv_writer.writerow([tts[start_idx:end_idx], srcSig[start_idx:end_idx, :]])

bids_ieeg_run = f'{pid_bids}_{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg'
print('Saving: ' + bids_ieeg_run)
seeg_save = np.ascontiguousarray(seeg[:, start_idx:end_idx])
# param_desc = f"{choi.split('-')[0]} -> [+] {choi.split('-')[0][:-1]}{choi[-1]} -> [-] Impulsions biphasiques : {int(freq)}Hz | {int(I)}mA | Duree : {int(n_sec)}s"
# print(param_desc)
# events = [{"onset":int(onset), "type":"Comment", "duration":int(stim_length), "channels":choi, "description":param_desc}]
write_brainvision(data=np.ndarray(seeg_save.shape, buffer=seeg_save), sfreq=sfreq, ch_names=bip.ch_names,
                  fname_base=bids_ieeg_run, folder_out=f'{save_data}/{pid_bids}/{ses}/ieeg', overwrite=True)
                 # events=events

# saving channel description
bids_channels_run = f'{pid_bids}_{ses}_task-{task}_run-0{run}_channels.tsv'
print('Saving: ' + bids_channels_run)
with open(f'{save_data}/{pid_bids}/{ses}/ieeg/{bids_channels_run}', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(
        ['name', 'type', 'units', 'low_cutoff', 'high_cutoff', 'reference', 'group', 'sampling_frequency',
         'description', 'notch', 'status', 'status_description'])
    for ch in bip.ch_names:
        status = "good"
        if ch in bip.info['bads']:
            status = "bad"
        tsv_writer.writerow(
            [ch, 'SEEG', 'microV', 'n/a', 'n/a', 'n/a', ch[0], str(1000), 'SEEG', 'n/a', status, 'bipolar'])

# save relevant figures
save_fig = True
# now rerun al the figures and they'll save automatically


# ## 6.b Save these data only once per patient

# save connectome and gain matrix (the same as in tvb folder, just renamed)
print(f'Save {save_struct}/{pid_bids}_connectome.zip')
get_ipython().system(' cp {subj_proc_dir}/tvb/connectivity.vep.zip {save_struct}/{pid_bids}_connectome.zip')

print(f'Save {save_struct}/{pid_bids}_gain.txt')
get_ipython().system(' cp {subj_proc_dir}/elec/gain_inv-square.vep.txt {save_struct}/{pid_bids}_gain.txt')

# save sources and sensors image in 3D space
print(f'Save {save_struct}/img/{pid_bids}_sources_sensors.pgn')
get_ipython().system(' cp {subj_proc_dir}/elec/elec.vep.png {save_struct}/img/{pid_bids}_sources_sensors.png')

# In[219]:


seeg_coord = []
seeg_names = []
with open(f'{subj_proc_dir}/elec/seeg.xyz', 'r') as fd:
    for line in fd.readlines():
        # remove the line terminator for each line
        line = line[:-1]
        # take the coordinates
        res = [w for w in line.split(' ') if w != '']
        seeg_coord.append([float(res[1]), float(res[2]), float(res[3])])
        seeg_names.append(res[0])
seeg_coord = np.asarray(seeg_coord)
seeg_names = np.asarray(seeg_names)

# saving sources coordinates and name of each source
with open(f'{save_data}/{pid_bids}/{pid_bids}_electrodes.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['name', 'x', 'y', 'z', 'group',
                         'hemisphere'])  # should I also add here hemisphere info and size and material as present in the original _electrodes.tsv data ???
    for idx in range(seeg_names.shape[0]):
        if "'" in seeg_names[idx]:
            hemi = "L"
        else:
            hemi = "R"
        tsv_writer.writerow(
            [seeg_names[idx], seeg_coord[idx][0], seeg_coord[idx][1], seeg_coord[idx][2], seeg_names[idx][0], hemi])

# In[292]:


# TODO I think this is not necessary, maybe just put it in the root, cause it's the same for all patients
# TODO maybe just delete it for all patients... it doesn't seem useful
# with open(f'{save_data}/{pid_bids}/{pid_bids}_coordsystem.json', 'w') as json_file:
#     data = {"iEEGCoordinateSystem":"T1w",
#             "iEEGCoordinateUnits":"vox",
#             "iEEGCoordinateProcessingDescription":"SEEG contacts segmentation on CT scan",
#             "IntendedFor":"",
#             "iEEGCoordinateProcessingReference":"Medina et al. 2018 JNeuroMeth"
#            }
#     json.dump(data, json_file)


# # Do this once for the whole dataset

# In[132]:


with open(f'{save_data}/dataset_description.json', 'w') as json_file:
    #     data = json.load(json_file)
    data = {
        "Name": "Virtual Epileptic Patient Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "Authors": ["Borana Dollomaja", "Huifang Wang"],
        "GeneratedBy": [
            {
                "Name": "Semi-Automatic",
                "Description": "Added model parameter values used to generate the synthetic SEEG data, alongside SC and Gain matrices."
            }
        ],
        "HowToAcknowledge": "Please cite this paper: https://XXXX",
        "Funding": [
            "Human Brain Project Grant NR XXXX"
        ],
        "ReferencesAndLinks": [
            "https:XXXX",
            "XXXX"
        ],
        "DatasetDOI": "doi:XXXXX"
    }
    json.dump(data, json_file)

# In[156]:


# convert VEP atlas in tsv format
with open(f'{save_data}/VepMrtrixLut.txt', 'r') as fd:
    lines = fd.readlines()

with open(f'{save_data}/vep_atlas.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['label', 'region_name', 'r', 'g', 'b', 't'])
    for line in lines:
        i, roi_name, r, g, b, t = line.strip().split()
        tsv_writer.writerow([i, roi_name, r, g, b, t])

# In[189]:


# saving participants file
with open(f'{save_data}/participants.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['participant_id'])
    for i in range(50):
        tsv_writer.writerow(['sub-{0:03}'.format(i + 1)])

# In[ ]:




