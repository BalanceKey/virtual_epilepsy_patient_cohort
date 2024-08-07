'''
3D plots of SEEG electrodes over patient's brain surface
'''
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import pandas as pd
import colorednoise as cn
import sys
import mne
import os
sys.path.insert(1, '/Users/dollomab/OtherProjects/epi_visualisation/')
sys.path.insert(2, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
from util.utils import convert_to_pyvista_mesh, load_surfaces, load_bip_coords, plot_cortex_and_subcortex
from src.utils_simulate import  read_one_seeg_re_iis
from src.utils import get_ses_and_task, highpass_filter
import vep_prepare_ret

# TODO change
spikerate = False   # Plot spikerate, otherwise plot signal power
empirical = False  # Plot empirical data, otherwise plot simulated data
type_SEEG = 'stimulated' # ['spontaneous', 'interictal', 'stimulated']

assert (type_SEEG == 'spontaneous' and not spikerate) or (type_SEEG == 'interictal') or (type_SEEG == 'stimulated' and not spikerate)

#%% Select subject and data paths
pid = 'id008_dmc'#'sub-314a1ab2525d'#'id032_tc'#'id005_ft'#'id003_mg'
pid_bids = f'sub-{pid[2:5]}' #  'sub-314a1ab2525d'
subjects_dir = f'/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients/'
bids_path = '/Users/dollomab/MyProjects/Epinov_trial/simulate_data/patients/BIDS/VirtualEpilepticCohort'
util_dir = "/Users/dollomab/OtherProjects/epi_visualisation/util"
sub_dir_proc = f'{subjects_dir}/{pid}'

if spikerate:
    #%% Load computed spike rate for this patient
    hypothesis = 'VEPhypothesis'
    filepath_VEC = Path('~/MyProjects/Epinov_trial/simulate_data/patients_stats_emp_sim_interictal.csv')
    df = pd.read_csv(filepath_VEC)

    sub_df = df[df['subject_id'] == 'sub-008']
    for idx, sim_fname in zip(sub_df.index, sub_df['sim_interictal_fname']):
        if hypothesis in sim_fname:
            spikerate_emp_string = sub_df['emp_spike_rate_per_channel'][idx]
            spikerate_sim_string = sub_df['sim_spike_rate_per_channel'][idx]

    spikerate_emp_stringlist = spikerate_emp_string.strip("[]").split(' ')
    spikerate_emp_list = [el.strip("\n") for el in spikerate_emp_stringlist if el!='']
    spikerate_emp = pd.to_numeric(spikerate_emp_list)

    spikerate_sim_stringlist = spikerate_sim_string.strip("[]").split(' ')
    spikerate_sim_list = [el.strip("\n") for el in spikerate_sim_stringlist if el!='']
    spikerate_sim = pd.to_numeric(spikerate_sim_list)

    assert spikerate_sim.size == spikerate_emp.size
    if empirical:
        ch_names = sub_df['emp_ch_names'][sub_df.index[0]].strip("[]").split(', ')
        snsr_pwr = spikerate_emp
    else:
        ch_names = sub_df['sim_ch_names'][sub_df.index[0]].strip("[]").split(', ')
        snsr_pwr = spikerate_sim
    ch_names = [ch.replace('\"', '') for ch in ch_names]
    scale_N = 50
else:
    if empirical:
        if type_SEEG == 'interictal':
            #%% Load empirical SEEG data to plot
            szr_name = f'{sub_dir_proc}/seeg/fif/DMC_INTERC_161110B-DEX_0003.json'   # interictal data
            seeg_info, bip, gain, gain_prior = read_one_seeg_re_iis(sub_dir_proc, szr_name)  # load seizure data
            onset = 0  # plot ts_on sec before and after
            offset = 900                                            # taking 15 minutes here
            start_idx = int(onset * seeg_info['sfreq'])
            end_idx = int(offset * seeg_info['sfreq'])
        elif type_SEEG == 'spontaneous' or type_SEEG == 'stimulated':
            szr_name = f'{sub_dir_proc}/seeg/fif/DMC_criseStimB\'2-3_161115B-BEX_0003.json'  # spontaneous data
            # DMC_crise3_161114B-CEX_0006    DMC_crise2GS_161111M-AEX_0020  DMC_crise3_161114B-CEX_0006 DMC_criseStimB\'2-3_161115B-BEX_0003
            seeg_info, bip, gain, gain_prior = vep_prepare_ret.read_one_seeg_re(sub_dir_proc,
                                                                                szr_name)  # load seizure data
            base_length = 0  # plot ts_on sec before and after
            start_idx = int((seeg_info['onset'] - base_length) * seeg_info['sfreq'])
            base_length = 0
            end_idx = int((seeg_info['offset'] + base_length) * seeg_info['sfreq'])
        ch_names = bip.ch_names
        y = bip.get_data()[:, start_idx:end_idx]
        t = bip.times[start_idx:end_idx]
    else:
        # %% Loading synthetic SEEG data file
        clinical_hypothesis = False
        run = 1        # TODO change
        ses, task = get_ses_and_task(type=type_SEEG)
        if clinical_hypothesis:
            acq = "clinicalhypothesis"
        else:
            acq = "VEPhypothesis"
        print('ses ' + str(ses) + ' ' + task + ' ' + acq + ' run', run)
        sim_szr_name = f'{bids_path}/{pid_bids}/ses-0{ses}/ieeg/{pid_bids}_ses-0{ses}_task-{task}_acq-{acq}_run-0{run}_ieeg.vhdr'
        raw = mne.io.read_raw_brainvision(sim_szr_name, preload=True)
        ch_names_sim = raw.ch_names
        ch_names = ch_names_sim
        y_sim = raw._data
        y_sim_AC = highpass_filter(y_sim, 256, filter_order=101)
        t_sim = raw.times
        sfreq_sim = raw.info['sfreq']
        # Adding noise to y_sim
        beta = 1  # the exponent
        noise1 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
        beta = 2  # the exponent
        noise2 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
        beta = 3  # the exponent
        noise3 = cn.powerlaw_psd_gaussian(beta, y_sim.shape)
        # y_new = y_filt + noise + noise2
        y = y_sim_AC#y_sim_AC + noise1 * 4 + noise2 * 0.5

    # Compute electrode power from signal
    scale_N = 10#130
    snsr_pwr = (y**2).mean(axis=1)

energy_ch = (snsr_pwr-snsr_pwr.min())/(snsr_pwr.max()-snsr_pwr.min())*scale_N+0.000001

#%% Load cortical + subcortical surfaces
vep_subcort_aseg_file = f'{util_dir}/subcort.vep.txt'
vep_subcort_aseg = np.genfromtxt(vep_subcort_aseg_file, dtype=int)
py_vista_mesh, cort_parc = load_surfaces(sub_dir_proc, vep_subcort_aseg)  # Load pyvista mesh

#%% SEEG electrodes 3D plot
cmap = plt.cm.get_cmap("jet")
seeg_xyz_file = os.path.join(sub_dir_proc, "elec", "seeg.xyz")
lines = []
with open(seeg_xyz_file, 'r') as fd:
    for line in fd.readlines():
        name, *sxyz = line.strip().split()
        xyz = [float(_) for _ in sxyz]
        lines.append((name, xyz))

electrode_coord = np.genfromtxt(seeg_xyz_file, usecols=[1, 2, 3])
seeg_xyz_names = [label for label, _ in lines]
# do plotting
p = pv.Plotter(notebook=False)
p.set_background(color="white")
plot_cortex_and_subcortex(p, py_vista_mesh, vep_subcort_aseg, subcortex=True)

# plot electrodes
mesh_electrodes = pv.PolyData(electrode_coord)
glyphs = mesh_electrodes.glyph(geom=pv.Sphere(radius=1.5))
p.add_mesh(glyphs, show_scalar_bar=False, color=cmap(0)[0:3])
for ind_bip, ich_name in enumerate(ch_names):
    ch_name_monop = ich_name.rpartition('-')[0]    # get the first monopolar channel from bipolar ch_name
    if ch_name_monop[0] == "'":                     # just some error handling here when reading ch_names from csv file
        ch_name_monop = ch_name_monop[1:]
    ind_ch = seeg_xyz_names.index(ch_name_monop)
    #     p.add_mesh(pv.Sphere(radius=np.max([1.5,snsr_pwr[ind_bip]/3]),center=electrode_coord[ind_ch]),color=cmap(np.max([0,snsr_pwr[ind_bip]/10])))
    p.add_mesh(pv.Sphere(radius=np.max([1.5, np.log(energy_ch[ind_bip])]), center=electrode_coord[ind_ch]),
               color=cmap(np.max([0, energy_ch[ind_bip]]))[0:3], show_scalar_bar=True, opacity=1)
    # p.add_mesh(pv.Sphere(radius=np.max([1.5, energy_ch[ind_bip]]), center=electrode_coord[ind_ch]),
    #            color=cmap(np.max([0, energy_ch[ind_bip]]))[0:3], show_scalar_bar=True, opacity=1)

# _ = p.add_scalar_bar('Signal power (log scale)', vertical=True, # why is this colormap inaccurate ??
                       # title_font_size=35, label_font_size=30)
# p.update_scalar_bar_range(clim=[0,100])

# p.view_yz(negative=True)
p.view_xy(negative=False)
# p.view_xz(negative=True)
p.set_scale(xscale=1.3, yscale=1.3, zscale=1.3, reset_camera=False)
p.show(interactive=True, full_screen=False)
save_fig = False
if save_fig:
    save_name = f"{figure_dir}/SEEG_electrode_power_{pid}_simulated_seizure_run_{run}_{scale_N}_2.svg"
    print(save_name)
    p.save_graphic(save_name)

p.close()
