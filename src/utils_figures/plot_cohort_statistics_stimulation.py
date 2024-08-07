'''
Make figures of stimulation cohort statistics for paper
Esp. control location and control amplitude
'''
import pandas as pd
from pathlib import Path
from src.utils import compute_stats
import numpy as np
import matplotlib.pyplot as plt

hypothesis = 'VEPhypothesis'
filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated.csv')
filepath_Control = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated_control_amplitude.csv')

df = pd.read_csv(filepath_VEC)
df_control = pd.read_csv(filepath_Control)

# VEC cohort metrics
binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, \
agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = compute_stats(df, hypothesis)


#%% Control Amplitude cohort metrics

subjects = ['sub-002', 'sub-003', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011']
emp_stim_amplitudes = [2, 2.2, 2, 2, 2, 1.8]

metric_name = '2D_correlation'#'binary_overlap'#'2D_correlation'
metric_empirical = correlation_img_list
ytitle = 'Correlation'

stim_amp_control = [0.5, 1, 3, 4]
metric_control = []
for stim_amp in stim_amp_control:
    metric_values = df_control[df_control['stim_amplitude']==stim_amp][metric_name]
    metric_control.append(metric_values)

# Plot cohort statistics
boxplot=True

fig, ax = plt.subplots()
plt.title(f' All subjects - {hypothesis}', fontsize=18)
# Plotting all the data values

ax.scatter(np.zeros(shape=len(metric_control[0])), metric_control[0], color='lightblue')
ax.scatter(np.ones(shape=len(metric_control[1]))*1, metric_control[1], color='lightblue')

ax.scatter(np.ones(shape=len(metric_empirical))*2, metric_empirical, color='lightblue') # Empirical

ax.scatter(np.ones(shape=len(metric_control[2]))*3, metric_control[2], color='lightblue')
ax.scatter(np.ones(shape=len(metric_control[3]))*4, metric_control[3], color='lightblue')

if boxplot:
    # Plotting the mean values per column
    ax.scatter([0, 1, 2, 3, 4], [np.mean(metric_control[0]), np.mean(metric_control[1]), np.mean(metric_empirical),
                                 np.mean(metric_control[2]), np.mean(metric_control[3])], color='brown')

    # Plotting the median and quartiles (aka whiskerplot)... so professional
    ax.boxplot([ metric_control[0], metric_control[1], metric_empirical, metric_control[2], metric_control[3]],
               positions=np.array(range(5)))
else:
    ax.errorbar([0, 1, 2, 3, 4],
                [np.mean(metric_control[0]), np.mean(metric_control[1]), np.mean(metric_empirical),
                 np.mean(metric_control[2]), np.mean(metric_control[3])],
                [np.std(metric_control[0]), np.std(metric_control[1]), np.std(metric_empirical),
                 np.std(metric_control[2]), np.std(metric_control[3])], fmt='o', ecolor='steelblue',
                color='black', elinewidth=3, capsize=0)

plt.xticks([0, 1, 2, 3, 4], [f'{stim_amp_control[0]} mA', f'{stim_amp_control[1]} mA', 'Empirical \n (1.8-2.2 mA)', f'{stim_amp_control[2]} mA', f'{stim_amp_control[3]} mA'], fontsize=18)
plt.ylabel(ytitle, fontsize=18)
plt.tight_layout()
plt.show()

# TODO fix J_SP_Overlap cases