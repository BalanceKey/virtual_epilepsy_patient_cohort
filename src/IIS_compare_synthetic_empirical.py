''' Here we make summary figures out of the spike rate measurements between empirical and synthetic SEEG timeseries '''
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np

def compute_stats_for_hypothesis(df, hyp='VEPhypothesis'):
    '''
    Computes overal statistics on the dataframe 'df'
    for a certain hypothesis (VEPhypothesis or clinicalhypothesis)
    '''
    corr_list = []
    mse_list = []
    corr_grouped_list = []
    mse_grouped_list = []
    for i in range(1, 31):      # iterating each subject from 1 to 30
        sub = f'sub-0{i:02d}'
    # for sub in set(df['subject_id']):
        for row_id in df.loc[df['subject_id'] == sub].index:
            if hyp in df['sim_interictal_fname'][row_id]:
                correlation = df['correlation'][row_id]
                mse = df['mse'][row_id]
                correlation_grouped = df['grouped_correlation'][row_id]
                mse_grouped = df['grouped_mse'][row_id]

                corr_list.append(correlation)
                mse_list.append(mse)
                corr_grouped_list.append(correlation_grouped)
                mse_grouped_list.append(mse_grouped)
    return corr_list, mse_list, corr_grouped_list, mse_grouped_list

def plot_vec_vs_control(vec_stats, control_stats, axistitle, hyp):
    fig, ax = plt.subplots(figsize = (4,7))
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    plt.title(f'{hyp}', fontsize=18)
    ax.errorbar([0], [np.mean(vec_stats)],  [np.std(vec_stats)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='VEC IIS rate', transform=trans1)
    ax.scatter(np.zeros(shape=len(vec_stats)), vec_stats, color='lightblue', transform=trans1)

    ax.errorbar([0], [np.mean(control_stats)],  [np.std(control_stats)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='Control IIS rate', transform=trans2)
    ax.scatter(np.zeros(shape=len(control_stats)), control_stats, color='mistyrose', transform=trans2)

    plt.xticks([0], [axistitle], fontsize=18)
    plt.ylabel('Emp/Sim Correlation', fontsize=18)
    # plt.ylim([-1, 1])
    plt.tight_layout()
    plt.legend(fontsize=18, loc='best')
    plt.show()

def main():
    hyp = 'clinicalhypothesis' # TODO change
    # hyp =  'VEPhypothesis'

    # ------------------------ VEC Dataset --------------------------------------
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
    df = pd.read_csv(filepath)
    corr_list, mse_list, corr_grouped_list, mse_grouped_list = compute_stats_for_hypothesis(df, hyp=hyp)

    corr_list = np.array(corr_list)
    corr_grouped_list = np.array(corr_grouped_list)
    print(np.where(corr_list<-0.2))

    # ------------------------ Control Dataset --------------------------------------
    filepath = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal_control.csv')
    df_control = pd.read_csv(filepath)

    corr_list_c, mse_list_c, corr_grouped_list_c, mse_grouped_list_c = compute_stats_for_hypothesis(df_control, hyp=hyp)

    # Plotting results

    fig, ax = plt.subplots()
    plt.title(f'{hyp}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar([0, 1], [np.mean(corr_list), np.mean(corr_grouped_list)],  [np.std(corr_list), np.std(corr_grouped_list)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='VEC IIS rate', transform=trans1)
    ax.scatter(np.zeros(shape=len(corr_list)), corr_list, color='lightblue', transform=trans1)
    ax.scatter(np.ones(shape=len(corr_grouped_list)), np.asarray(corr_grouped_list), color='lightblue', transform=trans1)

    ax.errorbar([0, 1], [np.mean(corr_list_c), np.mean(corr_grouped_list_c)],  [np.std(corr_list_c), np.std(corr_grouped_list_c)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='ControlCohort IIS rate',
                transform=trans2)
    ax.scatter(np.zeros(shape=len(corr_list_c)), corr_list_c, color='mistyrose', transform=trans2)
    ax.scatter(np.ones(shape=len(corr_grouped_list_c)), np.asarray(corr_grouped_list_c), color='mistyrose', transform=trans2)

    plt.xticks([0, 1], ['Correlation', 'Correlation_grouped'], fontsize=18)
    plt.ylabel('Emp/Sim Correlation', fontsize=18)
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()

    fig, ax = plt.subplots()
    plt.title(f'{hyp}', fontsize=18)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar([0, 1], [np.mean(mse_list), np.mean(mse_grouped_list)], [np.std(mse_list), np.std(mse_grouped_list)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='VEC IIS rate', transform=trans1)
    ax.scatter(np.zeros(shape=len(mse_list)), mse_list, color='lightblue', transform=trans1)
    ax.scatter(np.ones(shape=len(mse_grouped_list)), np.asarray(mse_grouped_list), color='lightblue', transform=trans1)

    ax.errorbar([0, 1], [np.mean(mse_list_c), np.mean(mse_grouped_list_c)], [np.std(mse_list_c), np.std(mse_grouped_list_c)],
                fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, label='ControlCohort IIS rate',
                transform=trans2)
    ax.scatter(np.zeros(shape=len(mse_list_c)), mse_list_c, color='mistyrose', transform=trans2)
    ax.scatter(np.ones(shape=len(mse_grouped_list_c)), np.asarray(mse_grouped_list_c), color='mistyrose', transform=trans2)

    plt.xticks([0, 1], ['MSE', 'MSE_grouped'], fontsize=18)
    plt.ylabel('Emp/Sim ', fontsize=18)
    # plt.ylim([-1, 1])
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()

    plot_vec_vs_control(corr_list, corr_list_c, 'Correlation', hyp)

    plot_vec_vs_control(mse_list, mse_list_c, 'MSE', hyp)

    plot_vec_vs_control(corr_grouped_list, corr_grouped_list_c, 'Correlation grouped', hyp)

    plot_vec_vs_control(mse_grouped_list, mse_grouped_list_c, 'MSE grouped', hyp)
