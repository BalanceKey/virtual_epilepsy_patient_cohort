'''
Permutation testing to compare two samples (VEC vs control)
 H0 : mean(VEC) <= mean(Control)
 H1 : mean(VEC) > mean(Control)

 Also see the amazing example : https://www.jwilber.me/permutationtest/
'''
import numpy as np
from scipy.stats import t, ttest_ind
import matplotlib.pyplot as plt

# Permutation test
# NOTE: This assumes the ensembles have the same type of distribution
def p_val_from_permutation_test(vep_list, control_list, N=200000, plot_hist=True):
    N1 = len(vep_list)
    N2 = len(control_list)
    init_test_stat = np.mean(vep_list)-np.mean(control_list)
    grouped_list = vep_list+control_list
    # shuffle groups together N times
    normal_distrib = np.empty((N,))
    for i in range(N):
        np.random.shuffle(grouped_list)
        test_stat = np.mean(grouped_list[:N2]) - np.mean(grouped_list[N2:])
        normal_distrib[i] = test_stat
    if plot_hist:
        plt.figure()
        plt.hist(normal_distrib)
        plt.show()

    # calculate the p-value
    p_val = np.where(normal_distrib > init_test_stat)[0].size/N
    return p_val

# Other method : Using t-tests
# Computing p-value using t-tests
# H0 : mean(VEC) <= mean(Control)
# H1 : mean(VEC) > mean(Control)
# NOTE : This assumes also the samples have gaussian distribution
def p_value_from_t_test(vep_list, control_list):
    std_pooled = np.sqrt((np.std(vep_list)**2 + np.std(control_list)**2)/2)
    t_val = (np.mean(vep_list)-np.mean(control_list)) / (std_pooled*np.sqrt(1/len(vep_list) + 1/len(control_list)))
    dof = len(vep_list)+len(control_list)-2  # degrees of freedom
    # p-value for 2-sided t-test
    (2*(1 - t.cdf(abs(t_val), dof)))
    # p-value for 1-sided t-test
    (t.cdf(-abs(t_val), dof))
    p_value =  (1 - t.cdf(abs(t_val), dof))
    return p_value

# Other method : using scipy
# P-value calculation
# Calculates the T-test for the means of two independent samples of scores.
# This is a two-sided test for the null hypothesis that two independent samples have identical average (expected) values.
# This test assumes that the populations have identical variances by default.
# NOTE : This assumes also the samples have gaussian distribution
def p_value_from_scipy(vep_list, control_list):
    return ttest_ind(vep_list, control_list)


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"