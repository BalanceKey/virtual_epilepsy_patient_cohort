'''
Handles all the patients data
 - names the patients that have been simulated and the ones that haven't
 - keeps track of the data that has been generated
 - organises the data
 - visualises the data
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from src.utils import *
from src.data_statistics import get_metrics_for_subject, get_metric, plot_stim_metric_control_location

def create_dataframe_ebrains(save=True, N=30):
    ''' Creates dataframe for the fist time and saves it

        NSS : number of spontaneous seizures
        NIS: number of induced seizures
        NII: number of interictal spike recordings
    '''
    df = pd.DataFrame(columns=['ID', 'Sex', 'Age', 'Epilepsy type', 'MRI', 'Histopathology', 'Side', 'NSS', 'NIS', 'NII'])

    # subject ids
    subj_ids = []
    for i in range(N):
        subj_ids.append(f'sub-0{i+1:02d}')
    df['ID'] = subj_ids

    sexes = ['F', 'F', 'M', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'F', 'M',
           'M', 'M', 'M', 'M', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M',
           'F', 'F', 'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'F',
           'M', 'M', 'F', 'F', 'M']
    df['Sex'] = sexes[:N]
    ages = [34, 29, 36, 26,
            21, 59, 60, 46,
            45, 45, 41, 28,
            19, 23, 41, 29,
            26, 24, 44, 27,
            23, 30, 63, 29,
            42, 28, 29, 22,
            30, 56, 38, 36,
            23, 41, 31, 21,
            33, 33, 44, 35,
            26, 28, 23, 39,
            41, 32, 29, 22,
            22, 35]
    df['Age'] = ages[:N]
    # epilepsy_durations = [3, 10, 13, 3, 21, 14, 5, 8, 34, 18, 14, 9, 17, 18, 33, 23,
    #                     21, 22, 15, 10, 14, 23, 28, 15, 35, 24, 12, 14, 9, 45, 18,
    #                     21, 5, 8, 27, 13, 5, 22, 4, 19, 26, 19, 16, 39, 17, 31, 13,
    #                      7, 21, 21]
    # df['seizure_duration'] = seizure_durations
    epilepsy_types = [ 'Temporal', 'Temporo - occipital', 'Temporo - frontal', 'Temporal',
                      'Parietal', 'Frontal', 'Temporal', 'Temporal',
                      'Bifocal: parietal temporal', 'Temporal', 'Frontal', 'Bilateral temporo - frontal',
                      'Frontal', 'Premotor', 'Temporal', 'Temporo - fronto - parietal', 'Temporal',
                      'Parieto - temporal', 'Temporo - insular', 'Temporal', 'Occipital', 'Parietal', 'Temporal',
                      'Temporal', 'Insular', 'Occipital', 'Frontal', 'Temporo - frontal', 'Bilateral temporal',
                      'Temporo - frontal', 'Occipital', 'Bilateral temporal', 'Temporo-parietal',
                      'Temporal', 'Bilateral occipito-temporal', 'Temporo-insular', 'Temporal', 'Bilateral temporal',
                      'Temporo-frontal', 'Temporal mesial', 'Bilateral, temporo-parietal', 'Temporal',
                      'Premotor', 'Multifocal: parieto-premotor', 'Temporal',
                      'Insulo - parieto - premotor', 'Bilateral frontal', 'Premotor', 'Motor - opercular',
                      'Motor - premotor', 'Temporal', 'Prefronto-insular', 'Temporal']
    df['Epilepsy type'] = epilepsy_types[:N]

    MRIs= ['Normal',
        'L temporo - occipital PNH',
        'R temporo - occipital scar',
        'R temporal mesial ganglioglioma',
       'L postcentral - parietal gyration asymmetry',
        'Normal',
       'Normal',
        'L amygdala enlargement',
        'R parietal lesion',
        'L hippocampal sclerosis',
        'L frontal scar(abcess)',
        'Bilateral hippocampal and amygdala T2 - hypersignal',
    'Normal',
   ' Normal',
   'R temporal PMG and multiple PNH',
   'R temporo-parieto-insular and L temporo-parietal necrosis',
   'L temporo-polar hypothrophy and hippocampal sclerosis',
   'L Parieto - occipital necrosis (perinatal anoxy)',
    'Normal',
    'Normal',
    'Normal',
    'L parietal FCD',
   'Normal',
    'Normal',
   'Normal',
   'PNH',
    'R prefrontal gliotic scar (AVM)',
    'Anterior temporal necrosis',
    'Bilateral posterior PNH',
    'R Frontal FCD',
    'Normal',
    'L HH',
    'Normal',
    'Multiple R temporo - parietal PNH \ & temporal PMG',
    'R occipital mesial FCD',
    'R temporal anterior resection cavity',
    'R temporo-polar \& amygdala FCD, L post-chiasmal pilocytic astrocyrtoma',
    'Normal',
    'R fronto - temporal necrosis(gunshot injury)',
    'Hippocampal sclerosis',
    'R perisylvian  necrosis(perinatal stroke)',
    'Bilateral hippocampal sclerosis',
    'R precentral FCD',
    'L hippocampal and amygdala T2 hypersignal',
    'Normal',
    'Normal',
    'Normal',
    'R parietal DNET',
    'R fronto - opercular resection cavity',
    'L insulo - opercular necrosis(stroke)' ]
    df['MRI'] = MRIs[:N]

    histopathology = ['Hippocampal sclerosis', 'NA', 'FCD I', 'Ganglioglioma', 'NA', 'NA',
                      'mild gliosis', 'mild gliosis', 'mild gliosis', 'Hippocampal sclerosis', 'Gliosis',
                      'NA', 'mild gliosis', 'FCD IIb', 'NA', 'NA', 'Hippocampal sclerosis', 'NA', 'NA',
                      'Hippocampal sclerosis', 'FCD Ic', 'FCD IIb', 'NA', 'NA', 'NA', 'NA', 'Gliosis',
                      'Gliosis', 'NA', 'FCD IIb', 'NA', 'NA', 'Hippocampal sclerosis', 'NA', 'NA',
                      'Gliosis', 'FCD IIb', 'NA', 'Gliosis', 'Hippocampal sclerosis', 'NA', 'NA',
                      'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'FCD Ic', 'FCD IIa', 'FCD IIa']
    df['Histopathology'] = histopathology[:N]

    side = ['R', 'L', 'R', 'R', 'L', 'L', 'R>L', 'L', 'R', 'L', 'L', 'R&L', 'L', 'L', 'R', 'R>L', 'L', 'L', 'L>R', 'R',
            'L', 'L', 'L', 'R', 'L', 'R', 'R>L', 'R', 'R>L', 'R', 'R', 'R&L', 'R', 'R', 'R&L', 'R', 'R', 'L&R', 'R',
            'L', 'R>L', 'L', 'R', 'L', 'L', 'R', 'R&L', 'R', 'R', 'L', 'R', 'L', 'L']
    df['Side'] = side[:N]

    spontaneous_szrs = np.zeros(N)
    induced_szrs = np.zeros(N)

    patients_manager_path = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
    patients_df = pd.read_csv(patients_manager_path)
    for i, seizures in enumerate(patients_df['sim_spontaneous']):
        if i == N:
            break
        if not pd.isnull(seizures):
            spontaneous_szrs[i] = len(seizures.split(','))

    for i, seizures in enumerate(patients_df['sim_stimulated']):
        if i == N:
            break
        if not pd.isnull(seizures):
            seizure_list = seizures.split(',')
            if seizure_list[0] == '[]':
                seizure_len = 0
            else:
                seizure_len = len(seizure_list)
            induced_szrs[i] = seizure_len

    df['NSS'] = spontaneous_szrs
    df['NIS'] = induced_szrs
    df['NII'] = np.ones(N)

    # save dataframe
    if save:
        filepath = Path('~/MyProjects/Epinov_trial/simulate_data/ebrains_subject_data_paper.xlsx')
        df.to_excel(filepath)

def create_dataframe(save=True):
    ''' Creates dataframe for the fist time and saves it '''
    df = pd.DataFrame(columns=['subject_id', 'seizures', 'sim_spontaneous', 'sim_stimulated', 'sim_interictal','done'])

    # subject ids
    subj_ids = []
    for i in range(50):
        subj_ids.append(f'id0{i+1:02d}')

    df['subject_id'] = subj_ids
    df['done'] = np.repeat(0, 50)  # nothing is done for now

    # save dataframe
    if save:
        filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
        df.to_csv(filepath, index=False)

def update_patient_seizures():
    ''' Updates existing dataframe with all seizure recordings each patient has '''
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
    df = pd.read_csv(filepath)

    # for each patient add their seizures to the database
    for patient_index in [28]:
        subj_dir = Path(f'~/MyProjects/Epinov_trial/retrospective_patients/id0{patient_index+1:02d}*')
        subj_seizures = f'{subj_dir}/seeg/fif/'

        status = subprocess.run(f'ls {subj_seizures}/*.json', shell=True, capture_output=True)
        out = status.stdout.decode('utf-8').strip().split('\n')
        seizures = [s.split('/')[-1].split('.')[0] for s in out]
        df.at[patient_index, 'seizures'] = seizures

    save=True
    if save:
        filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
        df.to_csv(filepath, index=False)

def update_seizures_to_simulate():
    ''' Updates seizures that need to be simulated (seizures of same type are only simulated once) '''

    patient_index = 11

    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
    df = pd.read_csv(filepath)
    subj_dir = Path(f'~/MyProjects/Epinov_trial/retrospective_patients/id0{patient_index + 1:02d}*')
    subj_seizures = f'{subj_dir}/seeg/fif/'
    status = subprocess.run(f'ls {subj_seizures}/*.json', shell=True, capture_output=True)
    out = status.stdout.decode('utf-8').strip().split('\n')
    seizures = [s.split('/')[-1].split('.')[0] for s in out]

    # df['sim_spontaneous'] = df['sim_spontaneous'].astype(object) # to fix error when adding list inside a cell
    # df['sim_stimulated'] = df['sim_interictal'].astype(object)
    df.at[patient_index, 'sim_spontaneous'] = [seizures[0], seizures[1]]
    df.at[patient_index, 'sim_stimulated'] = []
    df.at[patient_index, 'sim_interictal'] = seizures[2]

    save=True
    if save:
        filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
        df.to_csv(filepath, index=False)

def check_patient_done(save=False):
    ''' Updates patients that are done !!! '''
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
    df = pd.read_csv(filepath)
    patient_index = 29
    df.at[patient_index, 'done'] = 1

    save=True
    if save:
        filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_manager.csv')
        df.to_csv(filepath, index=False)

def visualise_stats(filepath_VEC, filepath_Control, hypothesis='VEPhypothesis'):
    ''' Plots group statistics between VEC and Control datasets '''

    df = pd.read_csv(filepath_VEC)
    binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
    J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
    PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, \
    agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = \
        compute_stats(df, hyp=hypothesis, plot=False)

    df_control=pd.read_csv(filepath_Control)
    binary_overlap_listc, SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, J_SO_overlap_listc, \
    J_SP_overlap_listc, J_NS_overlap_listc, corr_env_listc, corr_signal_pow_listc, \
    PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc, \
    agreement_img_listc, correlation_img_listc, mse_img_listc, rmse_img_listc = compute_stats(df_control,
                                                                                              hyp=hypothesis,
                                                                                              plot=False)
    fig, ax = plt.subplots()
    plt.title(f'{hypothesis}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar([0, 1, 2, 3],
                [np.mean(correlation_img_list), np.mean(binary_overlap_list), np.mean(J_SO_overlap_list) / 100,
                 np.mean(J_SP_overlap_list) / 100],
                [np.std(correlation_img_list), np.std(binary_overlap_list), np.std(J_SO_overlap_list) / 100,
                 np.std(J_SP_overlap_list) / 100], fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0,
                transform=trans1, label='VEC')
    ax.scatter(np.zeros(shape=len(correlation_img_list)), correlation_img_list, transform=trans1, color='lightblue')
    ax.scatter(np.ones(shape=len(binary_overlap_list)), binary_overlap_list, transform=trans1, color='lightblue')
    ax.scatter(np.ones(shape=len(J_SO_overlap_list)) * 2, np.asarray(J_SO_overlap_list) / 100, transform=trans1,
               color='lightblue')
    ax.scatter(np.ones(shape=len(J_SP_overlap_list)) * 3, np.asarray(J_SP_overlap_list) / 100, transform=trans1,
               color='lightblue')

    ax.errorbar([0, 1, 2, 3],
                [np.mean(correlation_img_listc), np.mean(binary_overlap_listc), np.mean(J_SO_overlap_listc) / 100,
                 np.mean(J_SP_overlap_listc) / 100],
                [np.std(correlation_img_listc), np.std(binary_overlap_listc), np.std(J_SO_overlap_listc) / 100,
                 np.std(J_SP_overlap_listc) / 100], fmt='o', ecolor='tomato', color='black', elinewidth=3, capsize=0,
                transform=trans2, label='Control')
    ax.scatter(np.zeros(shape=len(correlation_img_listc)), correlation_img_listc, transform=trans2, color='mistyrose')
    ax.scatter(np.ones(shape=len(binary_overlap_listc)), binary_overlap_listc, transform=trans2, color='mistyrose')
    ax.scatter(np.ones(shape=len(J_SO_overlap_listc)) * 2, np.asarray(J_SO_overlap_listc) / 100, transform=trans2,
               color='mistyrose')
    ax.scatter(np.ones(shape=len(J_SP_overlap_listc)) * 3, np.asarray(J_SP_overlap_listc) / 100, transform=trans2,
               color='mistyrose')
    plt.xticks([0, 1, 2, 3], ['Correlation', 'Overlap', 'J_SO', 'J_SP'], fontsize=18)
    plt.ylabel('Emp/Sim Correlation', fontsize=18)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()

def treat_SP_special_case(SP_overlap_list):
    ''' Here treat the case where there is no SP in both emp and sim case
        by removing all these cases from the given list
        Basically we remove these cases from the stats here, but this is one way of many to handle this case '''
    new_SP_overlap_list = []
    for val in SP_overlap_list:
        if val >=0 :
            new_SP_overlap_list.append(val)
    return new_SP_overlap_list

# TODO finish and test this :
def compute_average_metrics_all_subjects_amp(df_vec, df_control, hypothesis='VEP_hypothesis'):
    ''' Computes overall metrics accross available subjects for both VEC and Control datasets '''

    subjects = ['sub-002', 'sub-003', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011']
    emp_stim_amplitudes = [2, 2.2, 2, 2, 2, 1.8]

    correlations_emp = []
    correlations_sim = []

    for sub in subjects:
        binary_overlap_list, correlation_img_list, J_SO_overlap_list, J_SP_overlap_list, agreement_img_list \
            = get_metrics_for_subject(df_control, hypothesis, sub, control_amp=False)

        binary_overlap_listc, correlation_img_listc, J_SO_overlap_listc, J_SP_overlap_listc, agreement_img_listc, \
        stim_amplitudes = get_metrics_for_subject(df_control, hypothesis, sub, control_amp=True)

        correlations_emp.append(correlation_img_list)
        correlations_sim.append(correlations_sim)

        # TODO continue this, too tired right now to do that...




def visualise_stats_by_group(filepath_VEC, filepath_Control, sub = 'sub-002', hypothesis='VEPhypothesis'):
    ''' Plots group statistics between VEC and Control datasets but groups them by location set
        Used specifically for Stim-ControlL-ocation dataset '''

    # Get metrics from VEC dataset

    df = pd.read_csv(filepath_VEC)
    binary_overlap_list, J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, \
    agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, \
    corr_env_list, corr_signal_pow_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, \
    PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list \
                                             = get_metrics_for_subject(df, hypothesis, sub)

    # Get metrics from control dataset

    df_control = pd.read_csv(filepath_Control)
    group_nr = 1
    binary_overlap_listc1, J_SO_overlap_listc1, J_SP_overlap_listc1, J_NS_overlap_listc1, \
    agreement_img_listc1, correlation_img_listc1, mse_img_listc1, rmse_img_listc1, \
    corr_env_listc1, corr_signal_pow_listc1, SO_overlap_listc1, SP_overlap_listc1, NS_overlap_listc1, \
    PCA_correlation_listc1, PCA1_correlation_listc1, PCA_correlation_slp_listc1, PCA1_correlation_slp_listc1 = \
        get_metrics_for_subject(df_control, hypothesis, sub, group_number=group_nr)

    group_nr = 2
    binary_overlap_listc2, J_SO_overlap_listc2, J_SP_overlap_listc2, J_NS_overlap_listc2, \
    agreement_img_listc2, correlation_img_listc2, mse_img_listc2, rmse_img_listc2, \
    corr_env_listc2, corr_signal_pow_listc2, SO_overlap_listc2, SP_overlap_listc2, NS_overlap_listc2, \
    PCA_correlation_listc2, PCA1_correlation_listc2, PCA_correlation_slp_listc2, PCA1_correlation_slp_listc2 = \
        get_metrics_for_subject(df_control, hypothesis, sub, group_number=group_nr)

    group_nr = 3
    binary_overlap_listc3, J_SO_overlap_listc3, J_SP_overlap_listc3, J_NS_overlap_listc3, \
    agreement_img_listc3, correlation_img_listc3, mse_img_listc3, rmse_img_listc3, \
    corr_env_listc3, corr_signal_pow_listc3, SO_overlap_listc3, SP_overlap_listc3, NS_overlap_listc3, \
    PCA_correlation_listc3, PCA1_correlation_listc3, PCA_correlation_slp_listc3, PCA1_correlation_slp_listc3 = \
        get_metrics_for_subject(df_control, hypothesis, sub, group_number=group_nr)

    group_nr = 4
    binary_overlap_listc4, J_SO_overlap_listc4, J_SP_overlap_listc4, J_NS_overlap_listc4, \
    agreement_img_listc4, correlation_img_listc4, mse_img_listc4, rmse_img_listc4, \
    corr_env_listc4, corr_signal_pow_listc4, SO_overlap_listc4, SP_overlap_listc4, NS_overlap_listc4, \
    PCA_correlation_listc4, PCA1_correlation_listc4, PCA_correlation_slp_listc4, PCA1_correlation_slp_listc4 = \
        get_metrics_for_subject(df_control, hypothesis, sub, group_number=group_nr)


    # Treating special SP case here
    yaxis_title = 'J_SP_Overlap'

    if yaxis_title == 'J_SP_Overlap':
        J_SP_overlap_list_new =  treat_SP_special_case(J_SP_overlap_list)
        J_SP_overlap_listc1_new = treat_SP_special_case(J_SP_overlap_listc1)
        J_SP_overlap_listc2_new = treat_SP_special_case(J_SP_overlap_listc2)
        J_SP_overlap_listc3_new = treat_SP_special_case(J_SP_overlap_listc3)
        J_SP_overlap_listc4_new = treat_SP_special_case(J_SP_overlap_listc4)

    feature_list_emp   = [J_SO_overlap_list[1]]#J_SO_overlap_list#agreement_img_list #J_SP_overlap_list_new #correlation_img_list #binary_overlap_list
    feature_list_simc1 = J_SO_overlap_listc1
    feature_list_simc2 = J_SO_overlap_listc2
    feature_list_simc3 = J_SO_overlap_listc3
    feature_list_simc4 = J_SO_overlap_listc4

    fig, ax = plt.subplots()
    plt.title(f' {sub} - {hypothesis}', fontsize=18)
    ax.errorbar([0, 1, 2, 3, 4],
                [np.mean(feature_list_emp), np.mean(feature_list_simc1), np.mean(feature_list_simc2),
                 np.mean(feature_list_simc3), np.mean(feature_list_simc4)],
                [np.std(feature_list_emp), np.std(feature_list_simc1), np.std(feature_list_simc2),
                 np.std(feature_list_simc3), np.std(feature_list_simc4)], fmt='o', ecolor='steelblue',
                color='black', elinewidth=3, capsize=0, label='VEC')
    ax.scatter(np.zeros(shape=len(feature_list_emp)), feature_list_emp, color='lightblue')
    ax.scatter(np.ones(shape=len(feature_list_simc1)), feature_list_simc1, color='lightblue')
    ax.scatter(np.ones(shape=len(feature_list_simc2))*2, feature_list_simc2, color='lightblue')
    ax.scatter(np.ones(shape=len(feature_list_simc3))*3, feature_list_simc3, color='lightblue')
    ax.scatter(np.ones(shape=len(feature_list_simc4))*4, feature_list_simc4, color='lightblue')

    plt.xticks([0, 1, 2, 3, 4], ['Empirical', 'Dist 1', 'Dist 2', 'Dist 3', 'Dist 4'], fontsize=18)
    plt.ylabel(yaxis_title, fontsize=18)
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()

def main():
    hypothesis = 'VEPhypothesis'
    filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated.csv')
    sub = 'sub-012'

    # Control Stim Location
    filepath_Control = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated_control_location.csv')
    visualise_stats_by_group(filepath_VEC=filepath_VEC, filepath_Control=filepath_Control, sub = sub, hypothesis=hypothesis)

    # Control Stim Amplitude
    filepath_ControlAmp = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated_control_amplitude.csv')

    # Get metrics from VEC dataset
    df = pd.read_csv(filepath_VEC)

    binary_overlap_list, J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, \
    agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, \
    corr_env_list, corr_signal_pow_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, \
    PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list \
                                             = get_metrics_for_subject(df, hypothesis, sub)


    # Get metrics from control dataset
    df_control = pd.read_csv(filepath_ControlAmp)

    stim_amplitudes, binary_overlap_listc, J_SO_overlap_listc, J_SP_overlap_listc, J_NS_overlap_listc, \
    agreement_img_listc, correlation_img_listc, mse_img_listc,  rmse_img_listc, \
    corr_env_listc, corr_signal_pow_listc, SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, \
    PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc \
                                            = get_metrics_for_subject(df_control, hypothesis, sub, control_amp=True)

    # add the empirical stimulation amplitude (for plotting purposes)
    emp_stim_amp = 1# mA
    stim_amplitudes.append(emp_stim_amp)
    # arrange metrics by stimulation amplitude
    sorted_indexes = np.argsort(stim_amplitudes)

    yaxis_title = 'J_SP_overlap'

    if yaxis_title == 'J_SP_overlap':
        J_SP_overlap_list = treat_SP_special_case(J_SP_overlap_list)
        J_SP_overlap_listc = treat_SP_special_case(J_SP_overlap_listc)
    feature_list_emp = [J_SP_overlap_list[1]]
    feature_list_simc = J_SP_overlap_listc

    fig, ax = plt.subplots()
    plt.title(f' {sub} - {hypothesis}', fontsize=18)

    i = 0
    string_stim_amp = []
    for idx in sorted_indexes:
        if stim_amplitudes[idx] == emp_stim_amp:
            line1 = ax.scatter(np.ones(shape=len(feature_list_emp)) * i, feature_list_emp, color='steelblue', label='VEC')
            string_stim_amp.append(f'{emp_stim_amp}')
        else:
            line2 = ax.scatter(np.ones(shape=1)*i, feature_list_simc[idx], color='lightblue', label='Control')
            string_stim_amp.append(f'{stim_amplitudes[idx]}')
        i += 1

    plt.xticks(np.arange(len(string_stim_amp)), string_stim_amp, fontsize=18)
    plt.ylabel(yaxis_title, fontsize=18)
    plt.xlabel('mA', fontsize=18)
    plt.tight_layout()
    plt.legend(handles=[line1, line2], fontsize=18, loc='best')
    plt.show()



    # TODO change all stats for SP : when there is no propagation in both Emp and Simulated do not put it to 1, but to NaN !!!
    # TODO note: this change was done for the stim dataset, and stimcontrollocation and stimcontrolamplitude datasets only so far !


    #% Group averages control location

    df_control_loc = pd.read_csv(filepath_Control)

    metric_name = '2D_mse'
    yaxis_title = '2D_mse'
    correlations_emp = np.array(get_metric(df, hypothesis, metric_name))
    correlations_control_loc = []
    for group_nr in range(1,5):
        correlations = np.array(get_metric(df_control_loc, hypothesis, metric_name, group_number=group_nr))
        correlations_control_loc.append(correlations)

    plot_stim_metric_control_location(correlations_emp, correlations_control_loc, hypothesis, yaxis_title,
                                      boxplot=True)

    # TODO next: try violin plot
