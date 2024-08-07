'''
Checks for each patient it's data
 - the ieeg simulations
 - the derivated data
 - compares it to the empirical data (using signal power and spatio-temporal propagation)
'''
import mne
import matplotlib.pyplot as plt
import numpy as np
from mne_bids.copyfiles import copyfile_brainvision

def plot_seeg(seizure_name):
    raw = mne.io.read_raw_brainvision(seizure_name, preload=True)
    fig = plt.figure(figsize=(10, 20))
    scaleplt = 0.04
    ch_names = raw.ch_names
    y = raw._data
    for ind, ich in enumerate(ch_names):
        plt.plot(scaleplt * (y[ind, :] - y[ind, 0]) + ind, 'blue', lw=0.5);
    plt.xticks(fontsize=26)
    plt.ylim([-1, len(ch_names) + 0.5])
    # plt.xlim([t[0],t[-1]])
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=26)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.xlabel('Time', fontsize=50)
    plt.ylabel('Electrodes', fontsize=50)
    plt.title('SEEG recording', fontweight='bold', fontsize=50)
    plt.tight_layout()
    plt.show()

def main():
    database = f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort/'
    pid = 'id001_bt'
    pid_bids = f'sub-00{pid[-4]}'
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
    # plot_seeg(seizure_name)
    seizure_name2 = f'{sim_patient_data}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'

    copyfile_brainvision(seizure_name, seizure_name2, verbose=True)

    raw = mne.io.read_raw_brainvision(seizure_name, preload=True)
    fig = plt.figure(figsize=(10, 20))
    scaleplt = 0.04
    ch_names = raw.ch_names
    y = raw._data
    for ind, ich in enumerate(ch_names):
        plt.plot(scaleplt * (y[ind, :] - y[ind, 0]) + ind, 'blue', lw=0.5);
    plt.xticks(fontsize=26)
    plt.ylim([-1, len(ch_names) + 0.5])
    # plt.xlim([t[0],t[-1]])
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=26)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.xlabel('Time', fontsize=50)
    plt.ylabel('Electrodes', fontsize=50)
    plt.title('SEEG recording', fontweight='bold', fontsize=50)
    plt.tight_layout()
    plt.show()

