''' Here dataframes are analyzed and metrics are grouped and tailored statistics are computed '''

import matplotlib.pyplot as plt
import numpy as np

def get_metrics_for_subject(df, hypothesis, subject, group_number=None, control_amp=False):
    ''' For a specific subject in a dataframe df, and a hypothesis, returns the list of computed metrics '''

    assert group_number is None or control_amp is False # Sanity check, these two conditions are independent

    if control_amp:
        control_amplitudes_list = []

    binary_overlap_list = []
    corr_env_list = []
    corr_signal_pow_list = []

    J_SO_overlap_list = []
    J_SP_overlap_list = []
    J_NS_overlap_list = []

    SO_overlap_list = []
    SP_overlap_list = []
    NS_overlap_list = []

    PCA_correlation_list = []
    PCA1_correlation_list = []
    PCA_correlation_slp_list = []
    PCA1_correlation_slp_list = []

    agreement_img_list = []
    correlation_img_list = []
    mse_img_list = []
    rmse_img_list = []

    # select rows corresponding to a specific subject
    for row_id in df.loc[df['subject_id'] == subject].index:
        # select values from a specific EZ hypothesis
        if hypothesis in df['sim_seizure'][row_id]:
            # make sure the group field exists
            assert (group_number is None) or (group_number is not None and 'group' in df.keys())
            # if there's a group specified, select only values from that group
            if (group_number is None) or (group_number is not None and df['group'][row_id] == group_number):

                binary_overlap = df['binary_overlap'][row_id]
                corr_env = df['corr_envelope_amp'][row_id]
                corr_signal_pow = df['corr_signal_pow'][row_id]

                J_SO_overlap = df['J_SO_overlap'][row_id]
                J_SP_overlap = df['J_SP_overlap'][row_id]
                J_NS_overlap = df['J_NS_overlap'][row_id]

                SO_overlap = df['SO_overlap'][row_id]
                SP_overlap = df['SP_overlap'][row_id]
                NS_overlap = df['NS_overlap'][row_id]

                PCA_correlation = df['PCA_correlation'][row_id]
                PCA1_correlation = df['PCA1_correlation'][row_id]
                PCA_correlation_slp = df['PCA_correlation_slp'][row_id]
                PCA1_correlation_slp = df['PCA1_correlation_slp'][row_id]

                agreement_img = df['2D_agreement'][row_id]
                correlation_img = df['2D_correlation'][row_id]
                mse_img = df['2D_mse'][row_id]
                rmse_img = df['2D_rmse'][row_id]

                binary_overlap_list.append(binary_overlap)
                corr_env_list.append(corr_env)
                corr_signal_pow_list.append(corr_signal_pow)

                J_SO_overlap_list.append(J_SO_overlap)
                J_SP_overlap_list.append(J_SP_overlap)
                J_NS_overlap_list.append(J_NS_overlap)

                SO_overlap_list.append(SO_overlap)
                SP_overlap_list.append(SP_overlap)
                NS_overlap_list.append(NS_overlap)

                PCA_correlation_list.append(PCA_correlation)
                PCA1_correlation_list.append(PCA1_correlation)
                PCA_correlation_slp_list.append(PCA_correlation_slp)
                PCA1_correlation_slp_list.append(PCA1_correlation_slp)

                agreement_img_list.append(agreement_img)
                correlation_img_list.append(correlation_img)
                mse_img_list.append(mse_img)
                rmse_img_list.append(rmse_img)

                if control_amp:
                    control_amplitudes_list.append(df['stim_amplitude'][row_id])

    if control_amp:
        return control_amplitudes_list, binary_overlap_list, J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, \
               agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, \
               corr_env_list, corr_signal_pow_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, \
               PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list

    else:
        return binary_overlap_list, J_SO_overlap_list, J_SP_overlap_list, J_NS_overlap_list, \
               agreement_img_list, correlation_img_list, mse_img_list, rmse_img_list, \
               corr_env_list, corr_signal_pow_list, SO_overlap_list, SP_overlap_list, NS_overlap_list, \
               PCA_correlation_list, PCA1_correlation_list, PCA_correlation_slp_list, PCA1_correlation_slp_list

def get_metric(df, hypothesis, metric_name, group_number=None):
    ''' For a specific hypothesis in a dataframe df,
        returns all values for a metric specified in metric_name (aka a column name)
        [optional] if group_number is defined, returns metric values for that group number only '''

    assert metric_name in df  # checking the metric name exists in the dataframe
    if group_number is None:
        return df[metric_name]
    else:
        return df[df['group'] == group_number][metric_name]

def plot_stim_metric_control_location(metric_empirical, metric_control, hypothesis, ytitle, boxplot=True):

    fig, ax = plt.subplots()
    plt.title(f' All subjects - {hypothesis}', fontsize=18)

    # Plotting all the data values
    ax.scatter(np.zeros(shape=len(metric_empirical)), metric_empirical, color='lightblue')
    ax.scatter(np.ones(shape=len(metric_control[0])), metric_control[0], color='lightblue')
    ax.scatter(np.ones(shape=len(metric_control[1]))*2, metric_control[1], color='lightblue')
    ax.scatter(np.ones(shape=len(metric_control[2]))*3, metric_control[2], color='lightblue')
    ax.scatter(np.ones(shape=len(metric_control[3]))*4, metric_control[3], color='lightblue')

    if boxplot:
        # Plotting the mean values per column
        ax.scatter([0, 1, 2, 3, 4], [np.mean(metric_empirical), np.mean(metric_control[0]), np.mean(metric_control[1]),
                                     np.mean(metric_control[2]), np.mean(metric_control[3])], color='brown')

        # Plotting the median and quartiles (aka whiskerplot)... so professional
        ax.boxplot([metric_empirical, metric_control[0], metric_control[1], metric_control[2], metric_control[3]], positions=np.array(range(5)))
    else:
        ax.errorbar([0, 1, 2, 3, 4],
        [np.mean(metric_empirical), np.mean(metric_control[0]), np.mean(metric_control[1]),
        np.mean(metric_control[2]), np.mean(metric_control[3])],
        [np.std(metric_empirical), np.std(metric_control[0]), np.std(metric_control[1]),
         np.std(metric_control[2]), np.std(metric_control[3])], fmt='o', ecolor='steelblue',
        color='black', elinewidth=3, capsize=0)

    plt.xticks([0, 1, 2, 3, 4], ['Empirical', 'Dist 1', 'Dist 2', 'Dist 3', 'Dist 4'], fontsize=18)
    plt.ylabel(ytitle, fontsize=18)
    if ytitle == 'Correlation':
        plt.ylim([-0.2,1])
    plt.tight_layout()
    plt.show()
