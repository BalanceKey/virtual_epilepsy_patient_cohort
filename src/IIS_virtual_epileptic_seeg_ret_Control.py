'''
Generate control interictal spike data.
Done by taking the brain structure of one patient, and generating simulations with random EZ hypothesis from other
patients. Then map this onto SEEG of this patient and compute metrics again.
'''
# Importing libraries
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.image as mpimg
import os
import os.path as op
from pybv import write_brainvision
import scipy.signal as signal
import pandas as pd
import time as tm
import zipfile
import glob
import sys
import csv
import json
from src.utils_simulate import read_one_seeg_re_iis
from src.utils import isfloat

sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret
import vep_prepare

roi = vep_prepare_ret.read_vep_mrtrix_lut()

run = 3                     # TODO change
clinical_hypothesis = False # TODO change

ses = "ses-03"
task = "simulatedinterictalspikes"

# # 1. Importing patient specific information
# - Choosing a patient to work on.
# - Defining the folder to save the results for the patient.
# - Importing the structural connectivity, the list of regions from the parcellation.

pid = 'id005_ft'#'id004_bj'#'id003_mg' #'id001_bt'#'id030_bf'
pid_bids = f'sub-{pid[2:5]}'
subj_proc_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'

save_data = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/ControlCohort/'
os.makedirs(save_data, exist_ok=True)

try:
    if op.exists(f'{subj_proc_dir}/dwi_new/weights.txt'):
        SC = np.loadtxt(f'{subj_proc_dir}/dwi_new/weights.txt')
    else:
        print("taking connectivity from tvb")
        with zipfile.ZipFile(
                f'{subj_proc_dir}/tvb/connectivity.vep.zip') as sczip:
            with sczip.open('weights.txt') as weights:
                SC = np.loadtxt(weights)

except FileNotFoundError as err:
    print(f'{err}: Structural connectivity not found for {subj_proc_dir}')
SC[np.diag_indices(SC.shape[0])] = 0
SC = SC / SC.max()

# fig = plt.figure(figsize=(20, 20))
# plt.imshow(SC, norm=plc.LogNorm(vmin=1e-6, vmax=SC.max()))
# plt.title(f'{pid_bids}: Normalized SC (log scale)', fontsize=12, fontweight='bold')
# plt.xticks(np.r_[:len(roi)], roi, rotation=90)
# plt.yticks(np.r_[:len(roi)], roi)
# fig.tight_layout()
# save_sc = False
# if save_sc:
#     print('>> Save', f'{save_struct}/img/{pid_bids}_connectome.png')
#     plt.savefig(f'{save_struct}/img/{pid_bids}_connectome.png')
# plt.show()

# # 2. Choose a seizure to model
szr_name = f'{subj_proc_dir}/seeg/fif/FTinterC.json'
# Load seizure data
seeg_info, bip, gain, gain_prior = read_one_seeg_re_iis(subj_proc_dir, szr_name)
# plot ts_on sec before and after
onset = 0#800
start_idx = int(onset * seeg_info['sfreq'])
offset = 200#1800
end_idx = int(offset * seeg_info['sfreq'])
scaleplt = 0.004
y = bip.get_data()[:, start_idx:end_idx]
t = bip.times[start_idx:end_idx]
# load electrode positions
ch_names = bip.ch_names
fig = plt.figure(figsize=(40, 80))
nch_source = []
for ind_ch, ichan in enumerate(ch_names):
    isource = roi[np.argmax(gain[ind_ch])]
    nch_source.append(f'{isource}:{ichan}')
for ind, ich in enumerate(ch_names):
    plt.plot(t, scaleplt * (y[ind, :]) + ind, 'blue', lw=0.5);
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
#
# plt.imshow(mpimg.imread(f'{subj_proc_dir}/elec/elec.vep.png'))
# plt.show()
#
# nsensor, nchan = np.shape(gain)
# fig = plt.figure(figsize=(20, 20))
# im = plt.imshow(gain, norm=plc.LogNorm(vmin=gain.min(), vmax=gain.max()))
# plt.xticks(np.r_[:len(roi)], roi, rotation=90)
# plt.yticks(np.r_[:len(bip.ch_names)], bip.ch_names)
# plt.xlabel('Region#', fontsize=12)
# plt.ylabel('Channel#', fontsize=12)
# plt.title(f'{pid_bids}: Gain Matrix (log scale)', fontsize=12, fontweight='bold')
# plt.colorbar(im, fraction=0.03, pad=0.01)
# fig.tight_layout()
# save_gain = False
# if save_gain:
#     print('>> Save', f'{save_struct}/img/{pid_bids}_gain.png')
#     plt.savefig(f'{save_struct}/img/{pid_bids}_gain.png')
# plt.show()


# # 3. Setting up the simulation with TVB
# - Here we use the Epileptor
# - We set up the needed ingredients for the simulator (model, connectivity, coupling, integrator, initial conditions, monitors and more)

if clinical_hypothesis:
    acq = "clinicalhypothesis"
else:
    acq = "VEPhypothesis"

print(f'Simulation with noise using EZ from \"{acq}\".')

# We take parameters from the patient's saved simulation parameters
# We take the run 01 simulation by default
VEC_patient_data = f'{save_data}/../VirtualEpilepticCohort/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/'
simulator_parameters_path = f'{VEC_patient_data}/{pid_bids}_simulator_parameters_run-01.tsv'
epileptor_parameters_path = f'{VEC_patient_data}/{pid_bids}_epileptor_parameters_run-01.tsv'
simulator_parameters = pd.read_csv(simulator_parameters_path, sep='\t')
epileptor_parameters = pd.read_csv(epileptor_parameters_path, sep='\t')

slopestring = [el.strip('][') for el in epileptor_parameters['slope'][0].split(' ')]
slopearr = np.asarray([float(char) for char in slopestring if isfloat(char)])

initcondstring = [el.strip('][') for el in simulator_parameters['init_cond'][0].split(' ')]
initcondarr = np.asarray([float(char) for char in initcondstring if isfloat(char)])

noisecoeffsstring = [el.strip('][') for el in simulator_parameters['noise_coeffs'][0].split(' ')]
noisecoeffsarr = np.asarray([float(char) for char in noisecoeffsstring if isfloat(char)])

# We take x0 parameters from another random patient
epileptors = models.Epileptor(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'])
epileptors.r = np.array(epileptor_parameters['r'][0].strip('[]'), dtype=float) # Old method computed seizure length
num_regions = len(roi)
epileptors.slope = slopearr #np.ones(num_regions) * (0)
# epileptors.Iext = np.ones(num_regions) * (3.1) # these parameters never changed so we keep them the same
epileptors.Iext2 = np.ones(num_regions) * (0.45)
epileptors.Ks = np.ones(num_regions) * (1.0) * 0.001
epileptors.Kf = np.ones(num_regions) * (1.0) * 0.0001
epileptors.Kvf = np.ones(num_regions) * (1.0) * 0.001

period = float(simulator_parameters['period'])#1.
cpl_factor = float(simulator_parameters['coupling_factor'])#-0.05
dt = float(simulator_parameters['dt'])#0.05


# NOTE Here we take random ez hypothesis from other patient data
# We first check the EZ hypothesis used for the VEC patient, and then compare to random control
# To make sure they aren't identical

if clinical_hypothesis:
    print('Clinical Hypothesis')
    ezfile = json.load(open(f'{subj_proc_dir}/../ei-vep_53.json'))
    ez = ezfile[pid]['ez']
    print('EZ: ', ez)
else:
    print('VEP Hypothesis')
    # ezfile = np.loadtxt(f'{subj_proc_dir}/tvb/ez_hypothesis.vep.txt')
    ezfile = pd.ExcelFile(f'{subj_proc_dir}/../vep_pipeline_heatmap.xlsx', engine='openpyxl')

    # TODO change me back after this patient
    ezfile = pd.read_excel(ezfile, pid)
    # ezfile = pd.read_excel(ezfile, 'id029_bbc_i_ez_1')

    ezroi, ezval = np.array(ezfile['Brain Regions']), np.array(ezfile['Evs'])
    ez = [ezroi[iez] for iez in np.where(ezval >= 0.5)[0]]
    pz = [ezroi[iez] for iez in np.where((ezval >= 0.3) & (ezval < 0.5))[0]]
    pz2 = [ezroi[iez] for iez in np.where((ezval > 0) & (ezval < 0.3))[0]]
    print('EZ: ', ez)
    print('PZ: ', pz)
    print('PZ2: ', pz2)

samples = 30
rand_patient_id = np.random.choice(samples, size=1, replace=False)
rand_pid_bids = 'sub-'+'{:0>3}'.format(rand_patient_id[0]+1)
rand_patient_data = f'{save_data}/../VirtualEpilepticCohort/derivatives/tvb/{rand_pid_bids}/{ses}/{acq}/parameters/'
rand_epileptor_param_path = f'{rand_patient_data}/{rand_pid_bids}_epileptor_parameters_run-01.tsv'
rand_epileptor_parameters = pd.read_csv(rand_epileptor_param_path, sep='\t')
# TODO check this rand patient is not a patient we already simulated
assert rand_pid_bids != pid_bids
ez_rand = rand_epileptor_parameters['EZ'][0]
pz_rand = rand_epileptor_parameters['PZ'][0]
print(rand_pid_bids)
print(ez_rand)
print(pz_rand)

# Important note : Once we've checked the randEZ <> patientEZ we can use it
randx0string = [el.strip('][') for el in rand_epileptor_parameters['x0'][0].split(' ')]
randx0arr = np.asarray([float(char) for char in randx0string if isfloat(char)])
# Defining the excitability chosen from random patient
epileptors.x0 = randx0arr
# Defining Iext for these regions
randx0string = [el.strip('][') for el in rand_epileptor_parameters['Iext'][0].split(' ')]
randIextarr = np.asarray([float(char) for char in randx0string if isfloat(char)])
epileptors.Iext = randIextarr


# Connectivity
con = connectivity.Connectivity.from_file(str(subj_proc_dir + '/tvb/connectivity.vep.zip'))
con.tract_lengths = np.zeros((con.tract_lengths.shape))  # no time-delays
con.weights[np.diag_indices(con.weights.shape[0])] = 0
# con.weights = np.log(con.weights+1)
con.weights /= con.weights.max()
con.configure()

# Coupling
coupl = coupling.Difference(a=np.array([cpl_factor]))

# Integrator
noisy_sim = True
if noisy_sim:
    #     hiss = noise.Additive(nsig = np.array([0.02, 0.02, 0., 0.0003, 0.0003, 0.]))
    #     nsig = np.array([0.005, 0.005, 0., 0.00005, 0.00005, 0.]) # little to no interictal spikes
    # nsig = np.array([0.02, 0.02, 0., 0.0001, 0.0001, 0.])  # little interictal spikes
    nsig = noisecoeffsarr #np.array([1.3, 1.3, 0., 0.002, 0.002, 0.])
    hiss = noise.Additive(nsig=nsig)
    hiss.ntau = 1  # for gaussian distributed coloured noise
    heunint = integrators.HeunStochastic(dt=dt, noise=hiss)
else:
    nsig = []
    heunint = integrators.HeunDeterministic(dt=dt)

# Monitors
mon_tavg = monitors.TemporalAverage(period=period)

# Initial conditions
init_cond = np.array(initcondarr)
print(init_cond)
init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))

# Simulator
sim = simulator.Simulator(model=epileptors,
                          initial_conditions=init_cond_reshaped,
                          connectivity=con,
                          coupling=coupl,
                          integrator=heunint,
                          conduction_speed=np.inf,
                          monitors=[mon_tavg])

sim.configure()

print("Starting simulation...")
tic = tm.time()
simulation_length=10000
ttavg = sim.run(simulation_length=simulation_length)
print("Finished simulation.")
print('execute for ', (tm.time() - tic) / 60.0, 'mins')
save_fig = False

# Save results in Control Cohort

# # 4. Visualizing the simulated data on the source level
tts = ttavg[0][0]
tavg = ttavg[0][1]
srcSig = tavg[:,0,:,0] - tavg[:,3,:,0]

start_idx = 0
end_idx = simulation_length
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
plt.title(f'{pid_bids}:{acq}')
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_source_timeseries_run-0{run}.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_source_timeseries_run-0{run}.png')
else:
    plt.show()

# # 5. Visualizing the data on the sensor level

seeg = np.dot(gain, srcSig.T)
if clinical_hypothesis:
    basicfilename = 'ClinicalHypothesis'
else:
    basicfilename = 'VEP_Hypothesis'
show_ch = bip.ch_names
sfreq = 250.
nch = [show_ch.index(ichan) for ichan in show_ch]
nch_sourse = []
for ind_ch, ichan in enumerate(show_ch):
    isource = roi[np.argmax(gain_prior[ind_ch])]
    nch_sourse.append(f'{isource}:{ichan}')
plt.figure(figsize=[40, 70])
scaleplt = 0.02
base_length = int(5 * sfreq)
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (seeg[ich, start_idx:end_idx] - seeg[ich, 0]) + ind, 'blue', lw=1)
plt.xticks(fontsize=18)
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
else:
    plt.show()

# Applying a high pass filter to the signal to resseble AC recordings

def highpass_filter(y, sr):
    """In this case, the filter_stop_freq is that frequency below which the filter MUST act like a stop filter and
       filter_pass_freq is that frequency above which the filter MUST act like a pass filter.
       The frequencies between filter_stop_freq and filter_pass_freq are the transition region or band."""
    filter_stop_freq = 3  # Hz
    filter_pass_freq = 3  # Hz
    filter_order = 501
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    # Apply high-pass filter
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

plt.figure(figsize=[40, 70])
scaleplt = 0.04
base_length = int(5 * sfreq)
y = highpass_filter(seeg, 256)  # seeg
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (y[ich, start_idx:end_idx] - y[ich, 0]) + ind, 'blue', lw=1);
# plt.xticks(fontsize=18)
plt.xticks([], [])
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
if save_fig:
    print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
    plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
else:
    plt.show()

# # 6. Saving the data in BIDS format
# Do this once you are happy with what you got.
#
# Once all the data has been generated, for one patient, it's time to convert it to BIDS format
#
# To check:
#
# bids-validator BIDS/VirtualCohortProject --config.ignore=99
#
# Notes :
# - each session corresponds to a recording type : spontaneous seizure, stimulated seizure or interictal spikes
# - each run corresponds to a different recording of the same session (e.g. another spontaneous seizure type)

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

# model and simulator parameters to run with TVB
print('Saving: ' + epi_params_file)
with open(f'{epi_params_file}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['rand_patient_id', 'rand_patient_name', 'EZ', 'PZ', 'x0', 'Iext', 'Iext2', 'slope', 'r', 'Ks', 'Kf', 'Kvf'])
    tsv_writer.writerow([rand_patient_id[0], rand_pid_bids,  ez_rand, pz_rand, epileptors.x0, epileptors.Iext, epileptors.Iext2, epileptors.slope, epileptors.r,
                         epileptors.Ks, epileptors.Kf, epileptors.Kvf])

print('Saving: ' + sim_params_file)
with open(f'{sim_params_file}.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['coupling_factor', 'noise_coeffs', 'init_cond', 'dt', 'period'])
    tsv_writer.writerow([cpl_factor, nsig, init_cond, dt, period])

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
write_brainvision(data=np.ndarray(seeg_save.shape, buffer=seeg_save), sfreq=1000, ch_names=bip.ch_names,
                  fname_base=bids_ieeg_run, folder_out=f'{save_data}/{pid_bids}/{ses}/ieeg',
                  events=None, overwrite=True)

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
