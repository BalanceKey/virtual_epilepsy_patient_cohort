
import json
import mne
import sys
import numpy as np
sys.path.insert(1, '/Users/dollomab/MyProjects/Epinov_trial/VEP_Internal_Science/fit/')
import vep_prepare_ret
from scipy.optimize import fsolve

def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)

    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x
def read_one_seeg_re_iis(subj_proc_dir, jsfname):
    ''' To read interictal recordings '''
    with open(jsfname, "r") as fd:
        js = json.load(fd)
    fifname = js['filename']
    raw = mne.io.Raw(f'{subj_proc_dir}/seeg/fif/{fifname}', preload=True)
    drops = [_ for _ in (js["bad_channels"] + js["non_seeg_channels"]) if _ in raw.ch_names]
    raw = raw.drop_channels(drops)
    basicfilename = jsfname.split('.json')[0]
    basicfilename = basicfilename.split('/seeg/fif/')[1]
    # read gain

    seeg_xyz = vep_prepare_ret.read_seeg_xyz(subj_proc_dir)
    seeg_xyz_names = [label for label, _ in seeg_xyz]

    inv_gain_file = f'{subj_proc_dir}/elec/gain_inv-square.vep.txt'
    invgain = np.loadtxt(inv_gain_file)

    bip_gain_inv_minus, bip_xyz, bip_name = vep_prepare_ret.bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)
    bip_gain_inv_prior, _, _ = vep_prepare_ret.bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names, is_minus=False)
    # read the onset and offset

    bip = vep_prepare_ret._bipify_raw(raw)
    gain, bip = vep_prepare_ret.gain_reorder(bip_gain_inv_minus, bip, bip_name)
    gain_prior, _ = vep_prepare_ret.gain_reorder(bip_gain_inv_prior, bip, bip_name)
    seeg_info = {}
    seeg_info['fname'] = f'{basicfilename}'
    seeg_info['sfreq'] = bip.info['sfreq']

    try:
        seizure_onset = js['onset']
        seizure_offset = js['termination']
        seeg_info['onset'] = float(seizure_onset)
        seeg_info['offset'] = float(seizure_offset)

        print('Onset and offset information is available.')
    except:
        print('No onset or offset information available.')

    return seeg_info, bip, gain, gain_prior