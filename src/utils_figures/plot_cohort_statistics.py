'''
Make figures of cohort statistics for paper
'''
import pandas as pd
from pathlib import Path
from src.utils import compute_stats
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.transforms import Affine2D
import numpy as np
from src.permutation_test_comparison import p_val_from_permutation_test, p_value_from_t_test, p_value_from_scipy, convert_pvalue_to_asterisks


def plot_cohort_statistics(filepath_VEC, filepath_Control, hypothesis, boxplot=False, treat_SP_special_case=False):
    print('Using hypothesis ', hypothesis)

    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

    for i in range(2):
        if i == 0:
            df = pd.read_csv(filepath_VEC)
            trans = trans1
            color = 'lightblue'
            ecolor = 'steelblue'
            boxplotpos = -0.1
            label = 'VEC'# (N=16)'
        else:
            df = pd.read_csv(filepath_Control)
            trans = trans2
            color = 'mistyrose'
            ecolor = 'tomato'
            boxplotpos = 0.1
            label = 'RC'# (N=15)'

        binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
        J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
        PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, \
        agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = compute_stats(df, hypothesis)

        label += f' (N={len(binary_overlap_list)})'


        if treat_SP_special_case: # treating the case when J_SP_overlap = -100, aka no propagation in either empirical or spontaneous
            # J_SP_overlap_list_new = [val for val in J_SP_overlap_list if val >= 0]     # remove -100 cases
            J_SP_overlap_list_new = [abs(val) for val in J_SP_overlap_list]              # replace -100 by 100, not the best but the quick and dirty
            J_SP_overlap_list = J_SP_overlap_list_new

        # Plotting all data points
        ax.scatter(np.zeros(shape=len(correlation_img_list)), correlation_img_list, transform=trans, color=color, label=label)
        ax.scatter(np.ones(shape=len(binary_overlap_list)), binary_overlap_list, transform=trans, color=color)
        ax.scatter(np.ones(shape=len(J_SO_overlap_list)) * 2, np.asarray(J_SO_overlap_list) / 100, transform=trans,
                   color=color)
        ax.scatter(np.ones(shape=len(J_SP_overlap_list)) * 3, np.asarray(J_SP_overlap_list) / 100, transform=trans,
                   color=color)

        # Plotting averaged values and std
        if boxplot:
            # Plotting the mean values per column
            ax.scatter([0, 1, 2, 3],
                       [np.mean(correlation_img_list), np.mean(binary_overlap_list), np.mean(J_SO_overlap_list)/ 100,
                        np.mean(J_SP_overlap_list)/ 100], color='brown', transform=trans)

            # Plotting the median and quartiles (aka whiskerplot)... so professional
            ax.boxplot([correlation_img_list, binary_overlap_list, np.asarray(J_SO_overlap_list) / 100, np.asarray(J_SP_overlap_list) / 100],
                       positions=np.array(range(4)) + boxplotpos, widths = 0.15)
        else:
            ax.errorbar([0, 1, 2, 3],
                        [np.mean(correlation_img_list), np.mean(binary_overlap_list), np.mean(J_SO_overlap_list) / 100,
                         np.mean(J_SP_overlap_list) / 100],
                        [np.std(correlation_img_list), np.std(binary_overlap_list), np.std(J_SO_overlap_list) / 100,
                         np.std(J_SP_overlap_list) / 100], fmt='o', ecolor=ecolor, color='black', elinewidth=3,
                        capsize=0,
                        transform=trans)

    plt.xticks([0, 1, 2, 3], ['Correlation', 'Overlap', 'J_SO', 'J_SP'], fontsize=18)
    plt.ylabel('Emp/Sim Statistics', fontsize=18)
    plt.ylim([-0.2, 1.12])
    plt.tight_layout()
    plt.legend(fontsize=14, loc='lower center')
    plt.show()


#%% Spontaneous seizures

# hypothesis = 'VEPhypothesis'
hypothesis = 'clinicalhypothesis'
filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim.csv')
filepath_Control = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_control.csv')

plot_cohort_statistics(filepath_VEC, filepath_Control, hypothesis, boxplot=True)

#%% P-value Significance test
df = pd.read_csv(filepath_VEC)
df_control = pd.read_csv(filepath_Control)

binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, \
agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = compute_stats(df, hypothesis)

binary_overlap_listc, SO_overlap_listc, SP_overlap_listc, NS_overlap_listc, J_SO_overlap_listc, \
J_SP_overlap_listc, J_NS_overlap_listc, corr_env_listc, corr_signal_pow_listc, \
PCA_correlation_listc, PCA1_correlation_listc, PCA_correlation_slp_listc, PCA1_correlation_slp_listc, \
agreement_img_listc, correlation_img_listc, mse_img_listc, rmse_img_listc = compute_stats(df_control, hypothesis)

# Permutation test
p_val_correlation = p_val_from_permutation_test(correlation_img_list, correlation_img_listc)
p_val_overlap = p_val_from_permutation_test(binary_overlap_list, binary_overlap_listc)
p_val_SO = p_val_from_permutation_test(J_SO_overlap_list, J_SO_overlap_listc)
p_val_SP = p_val_from_permutation_test(J_SP_overlap_list, J_SP_overlap_listc)

# Print results with asterisks
print(f'P-value correlation: {p_val_correlation} {convert_pvalue_to_asterisks(p_val_correlation)}')
print(f'P-value overlap: {p_val_overlap} {convert_pvalue_to_asterisks(p_val_overlap)}')
print(f'P-value SO: {p_val_SO} {convert_pvalue_to_asterisks(p_val_SO)}')
print(f'P-value SP: {p_val_SP} {convert_pvalue_to_asterisks(p_val_SP)}')

# T-test
p_val_correlation = p_value_from_t_test(correlation_img_list, correlation_img_listc)
p_val_overlap = p_value_from_t_test(binary_overlap_list, binary_overlap_listc)
p_val_SO = p_value_from_t_test(J_SO_overlap_list, J_SO_overlap_listc)
p_val_SP = p_value_from_t_test(J_SP_overlap_list, J_SP_overlap_listc)


#############################################################################################################

#%% Stimulated seizures
# TODO add more metrics here, only 2 controls and only 10 stimulated seizures
hypothesis = 'clinicalhypothesis'
# hypothesis = 'VEPhypothesis'

filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated.csv')
filepath_Control = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_stimulated_control.csv')

plot_cohort_statistics(filepath_VEC, filepath_Control, hypothesis, boxplot=True, treat_SP_special_case=True)

df = pd.read_csv(filepath_VEC) # TODO add more metrics, only 10 for now!
binary_overlap_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, J_SO_overlap_list, \
J_SP_overlap_list, J_NS_overlap_list, corr_env_list, corr_signal_pow_list, \
PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list, \
agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list = compute_stats(df, hypothesis)
