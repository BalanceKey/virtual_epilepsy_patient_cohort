'''
Creating a control cohort based on stimulated seizure simulations. Taking the same parameters but changing here
systematically the stimulation amplitude parameter.

TODO since I'm really an ignorant human, I need to re-run my stimulations and then add the threshold parameter of the
epileptor on the derivative folder of that patient otherwise this work is not reproducible

Then, I need to change the stimulation I parameter and stimulation location a few times and hopefully get different results
'''

### PARENTHESE Looking at stim parameters across simulated patients
# import os.path
# import pandas as pd
# VEC_cohort_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort'
#
# stim_amplitudes = []
# for i in range(1, 31):
#     stim_params_path = f'{VEC_cohort_path}/derivatives/tvb/sub-{i:03d}/ses-02'
#     if os.path.isdir(stim_params_path):
#         stim_params_file = f'{stim_params_path}/VEPhypothesis/parameters/sub-{i:03d}_stimulation_parameters_run-01.tsv'
#         stim_params = pd.read_csv(stim_params_file, sep='\t')
#         stim_amplitude = stim_params['I'][0]
#         stim_amplitudes.append(stim_amplitude)
# #### END PARENTHESE

#!/usr/bin/env python
# Importing libraries
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn
import os.path as op
from pybv import write_brainvision
import scipy.signal as signal
import pandas as pd
import time as tm
import csv
from src.utils import isfloat
from src.utils_functions import vep_prepare_ret
from src.utils_functions.model import EpileptorStim
from src.utils_functions.model_2populations import EpileptorStim2Populations
from src.utils_functions.integrator import HeunDeterministicAdapted, HeunStochasticAdapted
roi = vep_prepare_ret.read_vep_mrtrix_lut()

clinical_hypothesis = False

# Applying a high pass filter to the signal to resseble AC recordings
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

# # 1. Importing patient specific information
# - Choosing a patient to work on.
# - Defining the folder to save the results for the patient.
# - Importing the structural connectivity, the list of regions from the parcellation.

pid = 'id012_fl'#'id009_ba'#'id008_dmc'#'id007_rd'  # 'id005_ft'#'id003_mg' # TODO change
pid_bids = f'sub-{pid[2:5]}'
subj_proc_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/{pid}'
save_data = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/ControlCohortStimAmplitude/'
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

# # 2. Choose a seizure to model
szr_name = f'{subj_proc_dir}/seeg/fif/FL_criseStimChocB3-4P_170222C-DEX_0008.json' # TODO change
# Load seizure data
seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(subj_proc_dir, szr_name)

# plot real data at sensor level with 5 sec before and after
plot = False
if plot:
    base_duration = 5
    scaleplt = 0.001
    f = vep_prepare_ret.plot_sensor_data(bip, gain_prior, seeg_info, base_duration, data_scaleplt=scaleplt,
                                     title=f'{pid}:ts_real_data')
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

print(f'Simulation {task} using EZ from \"{sub_folder}\".')

# ## 3b. Configure the stimulation
# - Select a pair of electrodes to apply the stimulation from
# - Calculate the effect of those electrodes' stimulation on the network nodes using the gain_prior matrix

# We take parameters from the patient's saved simulation parameters
# We take the run 01 simulation by default
VEC_patient_data = f'{save_data}/../VirtualEpilepticCohortStim/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/'
simulator_parameters_path = f'{VEC_patient_data}/{pid_bids}_simulator_parameters_run-01.tsv'
epileptor_parameters_path = f'{VEC_patient_data}/{pid_bids}_epileptor_parameters_run-01.tsv'
stimulation_parameters_path = f'{VEC_patient_data}/{pid_bids}_stimulation_parameters_run-01.tsv'
simulator_parameters = pd.read_csv(simulator_parameters_path, sep='\t')
epileptor_parameters = pd.read_csv(epileptor_parameters_path, sep='\t')
stimulation_parameters = pd.read_csv(stimulation_parameters_path, sep='\t')

# initcondstring = [el.strip('] [ \n') for el in simulator_parameters['init_cond'][0].split(' ')]
# initcondarr = np.asarray([float(char) for char in initcondstring if isfloat(char.strip('[]'))])
epileptor_ezstring = [el.strip('][') for el in epileptor_parameters['EZ'][0].split(', ')]
epileptor_ezarr = np.asarray([el.strip("'") for el in epileptor_ezstring])

epileptor_pzstring = [el.strip('][') for el in epileptor_parameters['PZ'][0].split(', ')]
epileptor_pzarr = np.asarray([el.strip("'") for el in epileptor_pzstring])

epileptor_x0string = [el.strip('][') for el in epileptor_parameters['x0'][0].split(' ')]
epileptor_x0arr =  np.asarray([float(char) for char in epileptor_x0string if isfloat(char)])
assert epileptor_x0arr.size == 162

epileptor_thresholdstring = [el.strip('][') for el in epileptor_parameters['threshold'][0].split(' ')]
epileptor_thresholdarr =  np.asarray([float(char) for char in epileptor_thresholdstring if isfloat(char)])
assert epileptor_thresholdarr.size == 162

stimulation_weights_string = [el.strip('[]') for el in stimulation_parameters['stimulation_weights'][0].split(' ')]
stimulation_weightsarr = np.asarray([float(char) for char in stimulation_weights_string if isfloat(char)])
assert stimulation_weightsarr.size == 162 # making sure there are as many weights as regions

epileptor_KSstring = [el.strip('][') for el in epileptor_parameters['Ks'][0].split(' ')]
epileptor_KSarr =  np.asarray([float(char) for char in epileptor_KSstring if isfloat(char)])
assert epileptor_KSarr.size == 162

epileptor_KFstring = [el.strip('][') for el in epileptor_parameters['Kf'][0].split(' ')]
epileptor_KFarr =  np.asarray([float(char) for char in epileptor_KFstring if isfloat(char)])
assert epileptor_KFarr.size == 162

epileptor_KVFstring = [el.strip('][') for el in epileptor_parameters['Kvf'][0].split(' ')]
epileptor_KVFarr =  np.asarray([float(char) for char in epileptor_KVFstring if isfloat(char)])
assert epileptor_KVFarr.size == 162

noise_coeffs_string = [el.strip('][') for el in simulator_parameters['noise_coeffs'][0].split(' ')]
noise_coeffs_arr = np.asarray([float(char) for char in noise_coeffs_string if isfloat(char)])

# Stimuli parameters
# freq = 50 # Hz
sfreq = float(simulator_parameters['sfreq']) # how many steps there are in one second
dt = float(simulator_parameters['dt'])#0.05
onset = int(stimulation_parameters['onset'])#2 * sfreq
# n_sec = 5 # stim duration
stim_length = float(stimulation_parameters['stimulation_length']) #n_sec * sfreq + onset # n_sec * sfreq + onset
simulation_length = float(simulator_parameters['simulation_length'])#50 * sfreq # n_sec * sfreq
T = float(stimulation_parameters['T']) # 1/freq * sfreq # pulse repetition period [ms]
tau = float(stimulation_parameters['pulse_width'])#2  # pulse duration [ms]
# I =  float(stimulation_parameters['I']) #2  # pulse intensity [mA]

# Here we take other I amplitudes for stimulation
run = 1
for I in [0.5, 2, 3, 4]: # TODO change
    print("Stimulation intensity : ", I)
    print("Run :", run)
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


    # Stimuli weighted accross the network nodes according to the gain matrix
    stim_weights = stimulation_weightsarr#(gain_prior[idx] - gain_prior[idx].min()) / (gain_prior[idx].max() - gain_prior[idx].min())
    stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                      connectivity=con,
                                      weight=stim_weights)
    stimulus.configure_space()
    stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))
    # plt.show()

    # epileptors = Epileptor3D4(variables_of_interest=['x1', 'y1', 'z', 'm'])
    epileptors = EpileptorStim2Populations(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'm'])
    epileptors.r = np.array([epileptor_parameters['r'][0].strip('[]')], dtype=float)#np.array([0.000025])#np.array([0.00002])
    r_val = float(epileptor_parameters['r2'][0].split(' ')[0].strip('[]'))#0.0004
    epileptors.r2 = np.ones(len(roi)) * (r_val)
    epileptors.Istim = np.ones(len(roi)) * (0.)
    epileptors.n_stim = np.zeros(len(roi))
    epileptors.Iext = np.ones(n_regions) * (3.1)
    epileptors.Iext2 = np.ones(n_regions) * (0.45)
    epileptors.Ks = epileptor_KSarr
    epileptors.Kf = epileptor_KFarr
    epileptors.Kvf = epileptor_KVFarr
    epileptors.threshold = epileptor_thresholdarr
    epileptors.x0 = epileptor_x0arr

    period = float(simulator_parameters['period'])
    cpl_factor = float(simulator_parameters['coupling_factor'])

    # Coupling
    coupl = coupling.Difference(a=np.array([cpl_factor]))

    # Integrator
    noisy_sim = True
    if noisy_sim:
        #     hiss = noise.Additive(nsig = np.array([0.02, 0.02, 0., 0.0003, 0.0003, 0.]))
        # nsig = np.array([0.001, 0.001, 0., 0.0005, 0.0005, 0., 0.]) # little to no interictal spikes
        nsig = noise_coeffs_arr# np.array([0.005, 0.005, 0., 0.0001, 0.0001, 0., 0.])
        hiss = noise.Additive(nsig=nsig)
        hiss.ntau = 1  # for gaussian distributed coloured noise
        heunint = HeunStochasticAdapted(dt=dt, noise=hiss)
    else:
        nsig = []
        heunint = HeunDeterministicAdapted(dt=dt)

    # Monitors
    mon_tavg = monitors.TemporalAverage(period=period)

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

    # get source level timeseries
    tts = ttavg[0][0]
    tavg = ttavg[0][1]
    srcSig = tavg[:, 0, :, 0] - tavg[:, 3, :, 0]
    start_idx = 0
    end_idx = int(simulation_length)
    srcSig_normal = srcSig / np.ptp(srcSig)

    # compute sensor level timeseries
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

    # # Saving the data in BIDS format
    #
    ses = "ses-02"
    task = "simulatedstimulation"

    if clinical_hypothesis:
        acq = "clinicalhypothesis"
    else:
        acq = "VEPhypothesis"
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

    epi_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_epileptor_parameters_run-0{run}'
    sim_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_simulator_parameters_run-0{run}'
    stim_params_file = f'{save_data}/derivatives/tvb/{pid_bids}/{ses}/{acq}/parameters/{pid_bids}_stimulation_parameters_run-0{run}'

    # model and simulator parameters to run with TVB
    print('Saving: ' + epi_params_file)
    with open(f'{epi_params_file}.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['EZ', 'PZ', 'x0', 'threshold', 'Iext', 'Iext2', 'r', 'r2', 'Ks', 'Kf', 'Kvf'])
        tsv_writer.writerow([epileptor_ezarr, epileptor_pzarr, epileptors.x0, epileptors.threshold, epileptors.Iext, epileptors.Iext2, epileptors.r, epileptors.r2,
                             epileptors.Ks, epileptors.Kf, epileptors.Kvf])

    print('Saving: ' + sim_params_file)
    with open(f'{sim_params_file}.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['coupling_factor', 'noise_coeffs', 'init_cond', 'dt', 'period', 'simulation_length', 'sfreq'])
        tsv_writer.writerow([cpl_factor, nsig, ic_full, dt, period, simulation_length, sfreq])

    choi = stimulation_parameters['stim_channels'][0]
    freq = 1/T*sfreq
    print('Saving: ' + stim_params_file)
    param_desc = f"{choi.split('-')[0]} -> [+] {choi.split('-')[0][:-1]}{choi[-1]} -> [-] Impulsions biphasiques : {freq}Hz | {I}mA |{stimulation_parameters['desc'][0].split('|')[-1]}"
    print(param_desc)
    with open(f'{stim_params_file}.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['stimulation_weights', 'stim_channels', 'stimulation_length', 'onset', 'T', 'I', 'pulse_width', 'desc'])
        tsv_writer.writerow([stim_weights, choi, stim_length, onset, T, I, tau, param_desc])

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

    events = [{"onset":onset, "type":"Comment", "duration":int(stim_length), "channels":choi, "description":param_desc}]
    write_brainvision(data=np.ndarray(seeg_save.shape, buffer=seeg_save), sfreq=sfreq, ch_names=bip.ch_names,
                      fname_base=bids_ieeg_run, folder_out=f'{save_data}/{pid_bids}/{ses}/ieeg',
                      events=events, overwrite=True)


    save_fig = True
    ## Saving images / Visualizing the simulated data on the source level
    #

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
    else:
        plt.show()

    # # 5. Visualizing the data on the sensor level

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
    else:
        plt.show()

    # filtering the sensor level data and adding noise to make it look more "realistic"
    plt.figure(figsize=[40, 70])
    scaleplt = 0.05
    # start_idx = 2100
    # end_idx = 7500#len(seeg)
    y = highpass_filter(seeg, 156)  # seeg
    beta = 1  # the exponent
    noise1 = cn.powerlaw_psd_gaussian(beta, y.shape)
    beta = 2  # the exponent
    noise2 = cn.powerlaw_psd_gaussian(beta, y.shape)
    beta = 4  # the exponent
    noise3 = cn.powerlaw_psd_gaussian(beta, y.shape)
    # y_new = y_filt + noise + noise2
    y_new = y + noise1 * 0.4 + noise2 * 0.2
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
    TS_on = start_idx + 5000
    TS_off = end_idx - 1850
    plt.axvline(TS_on, color='DeepPink', lw=2)
    plt.axvline(TS_off, color='DeepPink', lw=2)
    if save_fig:
        print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
        plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}_AC.png')
    else:
        plt.show()

    run += 1        # updating run number for the next simulation