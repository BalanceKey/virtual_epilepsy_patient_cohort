'''
Make figures of cohort statistics of interictal spikes for paper
'''
import pandas as pd
from pathlib import Path
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import numpy as np

from src.permutation_test_comparison import p_val_from_permutation_test, p_value_from_t_test, p_value_from_scipy, convert_pvalue_to_asterisks
from src.IIS_compare_synthetic_empirical import compute_stats_for_hypothesis

# hypothesis = 'VEPhypothesis'
hypothesis = 'clinicalhypothesis'

filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
filepath_Control = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal_control.csv')

df = pd.read_csv(filepath_VEC)
df_control = pd.read_csv(filepath_Control)

corr_list, mse_list, corr_grouped_list, mse_grouped_list = compute_stats_for_hypothesis(df, hyp=hypothesis)

corr_list_c, mse_list_c, corr_grouped_list_c, mse_grouped_list_c = compute_stats_for_hypothesis(df_control, hyp=hypothesis)

vec_stats = corr_grouped_list
control_stats = corr_grouped_list_c

boxplot = True

fig, ax = plt.subplots(figsize=(4, 6))
# fig, ax = plt.subplots(figsize=(4, 8))
trans1 = Affine2D().translate(0.0, 0.0) + ax.transData
trans2 = Affine2D().translate(+1.0, 0.0) + ax.transData
plt.title(f'{hypothesis}', fontsize=18)

ax.scatter(np.zeros(shape=len(vec_stats)), vec_stats, color='lightblue', transform=trans1, label='VEC')
ax.scatter(np.zeros(shape=len(control_stats)), control_stats, color='mistyrose', transform=trans2, label='Control')

if boxplot:
    # Plotting the mean values per column
    ax.scatter([0, 1], [np.mean(vec_stats), np.mean(control_stats)], color='brown')

    # Plotting the median and quartiles (aka whiskerplot)... so professional
    ax.boxplot([vec_stats, control_stats], positions=np.array(range(2)))
else:
    ax.errorbar([0], [np.mean(vec_stats)], [np.std(vec_stats)],
            fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0, transform=trans1)

    ax.errorbar([0], [np.mean(control_stats)], [np.std(control_stats)],
            fmt='o', ecolor='steelblue', color='black', elinewidth=3, capsize=0,
            transform=trans2)

plt.xticks([0, 1], [f'VEC\n(N={len(corr_list)})', f'RC\n(N={len(corr_list_c)})'], fontsize=18)
# plt.xticks([])
plt.ylabel('Emp/Sim IIS Correlation', fontsize=18)
# plt.ylim([-1, 1])
plt.tight_layout()
# plt.legend(fontsize=18, loc='lower left')
plt.show()


# P-value significance

# Permutation test
p_val_correlation = p_val_from_permutation_test(corr_list, corr_list_c)
p_val_correlation_gr = p_val_from_permutation_test(corr_grouped_list, corr_grouped_list_c)

# T-test
p_val_correlation = p_value_from_t_test(corr_list, corr_list_c)
p_val_correlation_gr = p_value_from_t_test(corr_grouped_list, corr_grouped_list_c)

# Print results with asterisks
print(f'P-value correlation: {p_val_correlation} {convert_pvalue_to_asterisks(p_val_correlation)}')
print(f'P-value correlation grouped: {p_val_correlation_gr} {convert_pvalue_to_asterisks(p_val_correlation_gr)}')
