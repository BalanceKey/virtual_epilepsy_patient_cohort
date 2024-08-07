#!/usr/bin/env python
# coding: utf-8

# Importing libraries
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.image as mpimg
import os.path as op
from pybv import write_brainvision
import scipy.signal as signal
import time as tm
import pandas as pd
import zipfile
import sys
import csv
import json
sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret
import vep_prepare
roi = vep_prepare_ret.read_vep_mrtrix_lut()

clinical_hypothesis = True
run = 2

# # 1. Importing patient specific information

# - Choosing a patient to work on.
# - Defining the folder to save the results for the patient.
# - Importing the structural connectivity, the list of regions from the parcellation.

pid = 'id030_bf'  # 'id005_ft'#'id003_mg'
pid_bids = f'sub-{pid[2:5]}'
subj_proc_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'

save_data = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'
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

fig = plt.figure(figsize=(20, 20))
plt.imshow(SC, norm=plc.LogNorm(vmin=1e-6, vmax=SC.max()))
plt.title(f'{pid_bids}: Normalized SC (log scale)', fontsize=12, fontweight='bold')
plt.xticks(np.r_[:len(roi)], roi, rotation=90)
plt.yticks(np.r_[:len(roi)], roi)
fig.tight_layout()
save_sc = False
if save_sc:
    print('>> Save', f'{save_struct}/img/{pid_bids}_connectome.png')
    plt.savefig(f'{save_struct}/img/{pid_bids}_connectome.png')
plt.show()

# # 2. Choose a seizure to model

szr_name = f'{subj_proc_dir}/seeg/fif/BF_crise2P_110831B-GEX_0001.json'
# Load seizure data
seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subj_proc_dir, szr_name)

# plot real data at sensor level with 5 sec before and after
base_duration = 10
scaleplt = 0.002
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

if clinical_hypothesis:
    acq = "clinicalhypothesis"
else:
    acq = "VEPhypothesis"

print(f'Simulation with noise using EZ from \"{acq}\".')

epileptors = models.Epileptor(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'])
seizure_length = seeg_info['offset'] - seeg_info['onset']
print(f"Seizure is {seizure_length} s long")
if seizure_length > 400:
    epileptors.r = np.array([0.00005])
elif 400 > seizure_length >= 110:
    epileptors.r = np.array([0.00015])
elif 110 > seizure_length >= 80:
    epileptors.r = np.array([0.0002])
elif 50 <= seizure_length < 80:
    epileptors.r = np.array([0.0003])
elif 30 < seizure_length < 50:
    epileptors.r = np.array([0.0004])
elif 10 < seizure_length < 30:
    epileptors.r = np.array([0.00055])
else:
    epileptors.r = np.array([0.0008])
num_regions = len(roi)
epileptors.slope = np.ones(num_regions) * (0) # TODO careful !!!!
epileptors.Iext = np.ones(num_regions) * (3.1)
epileptors.Iext2 = np.ones(num_regions) * (0.45)
epileptors.Ks = np.ones(num_regions) * (1.0) * 0.001
epileptors.Kf = np.ones(num_regions) * (1.0) * 0.0001  # ?
epileptors.Kvf = np.ones(num_regions) * (1.0) * 0.001  # ?

period = 1.
cpl_factor = -0.05
dt = 0.05
x0ez = -1.4 # TODO careful
x0pz = -1.98 # TODO careful
x0pz2 = -2.03 # TODO careful
x0num = -2.7
epileptors.x0 = np.ones(num_regions) * x0num

if clinical_hypothesis:
    print('Clinical Hypothesis')
    ezfile = json.load(open(f'{subj_proc_dir}/../ei-vep_53.json'))
    ez = ezfile[pid]['ez']
    pz = []
    pz2 = []

    print(ez)
    print(pz)
    print(pz2)

    # Getting the index for the ez & pz regions
    index_ez = [roi.index(iez) for iez in ez]
    index_pz = [roi.index(iez) for iez in pz]
    index_pz2 = [roi.index(iez) for iez in pz2]

    # Defining the excitability for those regions
    epileptors.x0[index_ez] = x0ez
    epileptors.x0[index_pz] = x0pz
    epileptors.x0[index_pz2] = x0pz2

else:
    # TODO NOTE here we take the entire heatmap values and try to map that onto x0 values (changed from patient id016)
    print('VEP Hypothesis')
    # ezfile = np.loadtxt(f'{subj_proc_dir}/tvb/ez_hypothesis.vep.txt')
    # if np.any(np.logical_and(ezfile != 0, ezfile != 1)):  #just to check if there are values greater than 0 other than 1
    #     print('>>>>>>>>>>> CAREFUL THERE ARE EZ VALUES IN ]0, 1[ RANGE !!!!!! <<<<<<<<<<<<<')
    # ez = [roi[iez] for iez in np.where(ezfile == 1)[0]]
    ezfile = pd.ExcelFile(f'{subj_proc_dir}/../vep_pipeline_heatmap.xlsx', engine='openpyxl')
    ezfile = pd.read_excel(ezfile, pid)
    ezroi, ezval = np.array(ezfile['Brain Regions']), np.array(ezfile['Evs'])
    # ez = [ezroi[iez] for iez in np.where(ezval >= 0.5)[0]]
    ez = [ezroi[iez] for iez in np.where(ezval > 0)[0]]
    values_of_interest = [ezval[iez] for iez in np.where(ezval > 0)[0]]
    index_ez = [roi.index(iez) for iez in ez]
    epileptors.x0[index_ez] = np.array(values_of_interest) * (2.2 - 1.2) - 2.2 # TODO careful

    pz = []
    pz2 = []
    print(ez)
    print(epileptors.x0[index_ez])
    print(pz)
    print(pz2)

    index_pz = [roi.index(iez) for iez in pz]
    index_pz2 = [roi.index(iez) for iez in pz2]
    epileptors.x0[index_pz] = x0pz
    epileptors.x0[index_pz2] = x0pz2

# epileptors.slope[index_pz] = -0.6

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
    nsig = np.array([0.005, 0.005, 0., 0.0001, 0.0001, 0.])
    hiss = noise.Additive(nsig=nsig)
    hiss.ntau = 1  # for gaussian distributed coloured noise
    heunint = integrators.HeunStochastic(dt=dt, noise=hiss)
else:
    nsig = []
    heunint = integrators.HeunDeterministic(dt=dt)

# Monitors
mon_tavg = monitors.TemporalAverage(period=period)

# epileptor_equil = models.Epileptor()
# epileptor_equil.x0 = np.array([-3.0])
init_cond = np.array(
    [-1.46242601e+00, -9.69344913e+00, 2.99029597e+00, -1.11181819e+00, -9.56105974e-20, -4.38727802e-01])
# init_cond = get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
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
simulation_length = 8000
ttavg = sim.run(simulation_length=simulation_length)
print("Finished simulation.")
print('execute for ', (tm.time() - tic) / 60.0, 'mins')
save_fig = False

# # 4. Visualizing the simulated data on the source level

tts = ttavg[0][0]
tavg = ttavg[0][1]
srcSig = tavg[:, 6, :, 0]  # tavg[:,0,:,0] - tavg[:,3,:,0]

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
plt.show()

start_idx = 0
end_idx = 1100
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
scaleplt = 0.04
base_length = int(5 * sfreq)
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (seeg[ich, start_idx:end_idx] - seeg[ich, 0]) + ind, 'blue', lw=1)
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
def highpass_filter(y, sr):
    """In this case, the filter_stop_freq is that frequency below which the filter MUST act like a stop filter and filter_pass_freq is that frequency above which the filter MUST act like a pass filter.
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
scaleplt = 0.05
base_length = int(5 * sfreq)
y = highpass_filter(seeg, 256)  # seeg
for ind, ich in enumerate(nch):
    plt.plot(tts[start_idx:end_idx], scaleplt * (y[ich, start_idx:end_idx] - y[ich, 0]) + ind, 'blue', lw=1)
# plt.xticks(fontsize=18)
plt.xticks([], [])
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([tts[start_idx], tts[end_idx - 1]])
plt.tight_layout()
plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
TS_on = start_idx + 100
TS_off = end_idx - 300
plt.axvline(TS_on, color='DeepPink', lw=2)
plt.axvline(TS_off, color='DeepPink', lw=2)
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
#
#
# Notes :
# - each session corresponds to a recording type : spontaneous seizure, stimulated seizure or interictal spikes
# - each run corresponds to a different recording of the same session (e.g. another spontaneous seizure type)
#

ses = "ses-01"
task = "simulatedseizure"

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
    tsv_writer.writerow(['EZ', 'PZ', 'x0', 'Iext', 'Iext2', 'slope', 'r', 'Ks', 'Kf', 'Kvf'])
    tsv_writer.writerow([ez, pz+pz2, epileptors.x0, epileptors.Iext, epileptors.Iext2, epileptors.slope, epileptors.r,
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


# ## 6.b Save these data only once per patient

# save connectome and gain matrix (the same as in tvb folder, just renamed)
print(f'Save {save_struct}/{pid_bids}_connectome.zip')
get_ipython().system(' cp {subj_proc_dir}/tvb/connectivity.vep.zip {save_struct}/{pid_bids}_connectome.zip')

print(f'Save {save_struct}/{pid_bids}_gain.txt')
get_ipython().system(' cp {subj_proc_dir}/elec/gain_inv-square.vep.txt {save_struct}/{pid_bids}_gain.txt')

# save sources and sensors image in 3D space
print(f'Save {save_struct}/img/{pid_bids}_sources_sensors.pgn')
get_ipython().system(' cp {subj_proc_dir}/elec/elec.vep.png {save_struct}/img/{pid_bids}_sources_sensors.png')

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


# saving participants file
with open(f'{save_data}/participants.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['participant_id'])
    for i in range(50):
        tsv_writer.writerow(['sub-{0:03}'.format(i + 1)])





