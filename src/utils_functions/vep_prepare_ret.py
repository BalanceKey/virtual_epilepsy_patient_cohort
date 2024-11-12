#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The codes provided in this file come from the VEP pipeline trial, they have been used to read and plot iEEG recordings
Original files are provided in the following reference:
Wang, Huifang E., et al. "Delineating epileptogenic networks using brain imaging data and personalized modeling
in drug-resistant epilepsy." Science Translational Medicine 15.680 (2023): eabp8982.

Created on Tue Oct  1 16:49:44 2019
@author: Huifang Wang  adapted from retrospective patients to trial patients
"""
import re
import os
import mne
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def seeg_ch_name_split(nm):
    """
    Split an sEEG channel name into its electrode name and index

    >>> seeg_ch_name_split('GPH10')
    ('GPH', 10)

    """

    try:
        elec, idx = re.match(r"([A-Za-z']+)(\d+)", nm).groups()
    except AttributeError as exc:
        return None
    return elec, int(idx)

def read_vep_mrtrix_lut(data_path = 'src/utils_data/VepMrtrixLut.txt'):
    roi_names = []
    with open(data_path, 'r') as fd:
        for line in fd.readlines():
            i, roi_name, *_ = line.strip().split()
            roi_names.append(roi_name)
            #roi_name_to_index[roi_name.lower()] = int(i) - 1
    roi=roi_names[1:]
    return roi


def bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names, is_minus=True):
    # from icdc import seeg_ch_name_split
    split_names = [seeg_ch_name_split(_) for _ in seeg_xyz_names]
    bip_gain_rows = []
    bip_xyz = []
    bip_names = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                if is_minus:
                    bip_gain_rows.append(gain[i + 1] - gain[i])
                else:
                    bip_gain_rows.append((gain[i + 1] + gain[i]) / 2.0)
                bip_xyz.append(
                    [(p + q) / 2.0 for p, q in zip(seeg_xyz[i][1], seeg_xyz[i + 1][1])]
                )
                bip_names.append("%s%d-%d" % (name, idx, next_idx))
        except Exception as exc:
            print(exc)
    # abs val, envelope/power always postive
    bip_gain = np.abs(np.array(bip_gain_rows))
    bip_xyz = np.array(bip_xyz)
    return bip_gain, bip_xyz, bip_names


def _bipify_raw(raw):
    # from icdc import seeg_ch_name_split
    split_names = [seeg_ch_name_split(_) for _ in raw.ch_names]
    bip_ch_names = []
    bip_ch_data = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                bip_ch_names.append('%s%d-%d' % (name, idx, next_idx))
                data, _ = raw[[i, i + 1]]
                bip_ch_data.append(data[1] - data[0])
        except:
            pass
    info = mne.create_info(
        ch_names=bip_ch_names,
        sfreq=raw.info['sfreq'],
        ch_types=['eeg' for _ in bip_ch_names])
    bip = mne.io.RawArray(np.array(bip_ch_data), info)
    return bip


def read_seeg_xyz(subj_proc_dir):
    lines = []
    fname = os.path.join(subj_proc_dir, 'elec/seeg.xyz')
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            name, *sxyz = line.strip().split()
            xyz = [float(_) for _ in sxyz]
            lines.append((name, xyz))
    return lines


def gain_reorder(bip_gain, raw, bip_names):
    gain_pick = []
    raw_drop = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_name in bip_names:
            gain_pick.append(bip_names.index(ch_name))
        else:
            raw_drop.append(ch_name)
    raw = raw.drop_channels(raw_drop)
    gain_pick = np.array(gain_pick)
    picked_gain = bip_gain[gain_pick]

    return picked_gain, raw

def read_one_seeg_re(subj_proc_dir, jsfname):  # mln
    with open(jsfname, "r") as fd:
        js = json.load(fd)

    fifname = js['filename']
    raw = mne.io.Raw(f'{subj_proc_dir}/seeg/fif/{fifname}', preload=True)
    drops = [_ for _ in (js["bad_channels"] + js["non_seeg_channels"]) if _ in raw.ch_names]
    raw = raw.drop_channels(drops)
    basicfilename = jsfname.split('.json')[0]
    basicfilename = basicfilename.split('/seeg/fif/')[1]

    # read gain

    seeg_xyz = read_seeg_xyz(subj_proc_dir)
    seeg_xyz_names = [label for label, _ in seeg_xyz]

    inv_gain_file = f'{subj_proc_dir}/elec/gain_inv-square.vep.txt'
    invgain = np.loadtxt(inv_gain_file)

    bip_gain_inv_minus, bip_xyz, bip_name = bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)
    bip_gain_inv_prior, _, _ = bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names, is_minus=False)
    # read the onset and offset

    seizure_onset = js['onset']

    seizure_offset = js['termination']

    bip = _bipify_raw(raw)
    gain, bip = gain_reorder(bip_gain_inv_minus, bip, bip_name)
    gain_prior, _ = gain_reorder(bip_gain_inv_prior, bip, bip_name)

    seeg_info = {}
    seeg_info['fname'] = f'{basicfilename}'
    seeg_info['onset'] = float(seizure_onset)
    seeg_info['offset'] = float(seizure_offset)
    seeg_info['sfreq'] = bip.info['sfreq']

    return seeg_info, bip, gain, gain_prior


def plot_sensor_data(bip, gain, seeg_info, ts_on, ts_cut=None, data_scaleplt=1, datafeature=None,
                     datafeature_scaleplt=0.7, title=None, figsize=[40, 70], yticks_fontsize=26, ch_selection=None,
                     data_baseline=False):
    # plot real data at sensor level
    show_ch = bip.ch_names
    nch_source = []
    roi = read_vep_mrtrix_lut()
    for ind_ch, ichan in enumerate(show_ch):
        isource = roi[np.argmax(gain[ind_ch])]
        nch_source.append(f'{isource}:{ichan}')

    # plot ts_on sec before and after
    base_length = int(ts_on * seeg_info['sfreq'])
    start_idx = int(seeg_info['onset'] * seeg_info['sfreq']) - base_length
    end_idx = int(seeg_info['offset'] * seeg_info['sfreq']) + base_length

    y = bip.get_data()[:, start_idx:end_idx]
    t = bip.times[start_idx:end_idx]

    if data_baseline:
        y -= np.mean(y[:, :base_length], axis=1, keepdims=True)

    # do same clipping as for datafeature in prepare_data_feature when plotting both
    if datafeature is not None and ts_cut is not None:
        cut_off_N = int(ts_cut * seeg_info['sfreq'])
        y = y[:, cut_off_N:]
        t = t[cut_off_N:]
        datafeature_t_start_idx = start_idx + cut_off_N
        datafeature_t_end_idx = datafeature_t_start_idx + datafeature.shape[0]

    f = plt.figure(figsize=figsize)

    if ch_selection is None:
        ch_selection = range(len(bip.ch_names))
    for ch_offset, ch_ind in enumerate(ch_selection):
        plt.plot(t, data_scaleplt * y[ch_ind] + ch_offset, 'blue', lw=0.5)
        if datafeature is not None:
            # plt.plot(t, datafeature_scaleplt*(datafeature.T[ch_ind]-datafeature[0,ch_ind])+ch_offset,'red',lw=1.5)
            plt.plot(bip.times[datafeature_t_start_idx:datafeature_t_end_idx],
                     datafeature_scaleplt * (datafeature.T[ch_ind] - datafeature[0, ch_ind]) + ch_offset, 'red', lw=1.5)

    vlines = [seeg_info['onset'], seeg_info['offset']]
    for x in vlines:
        plt.axvline(x, color='DeepPink', lw=3)

    # annotations
    if bip.annotations:
        for ann in bip.annotations:
            descr = ann['description']
            start = ann['onset']
            end = ann['onset'] + ann['duration']
            # print("'{}' goes from {} to {}".format(descr, start, end))
            if descr == 'seeg_bad_segments':
                plt.axvline(start, color='red', lw=1)
                plt.axvline(end, color='red', lw=1)

    plt.xticks(fontsize=18)
    plt.xlim([t[0], t[-1]])
    plt.yticks(range(len(ch_selection)), [nch_source[ch_index] for ch_index in ch_selection], fontsize=yticks_fontsize)
    plt.ylim([-1, len(ch_selection) + 0.5])

    plt.title(title, fontsize=16)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.tight_layout()

    return f


def seeg_log_power(a, win_len, pad=True):
    envlp = np.empty_like(a)
    # pad with zeros at the end to compute moving average of same length as the signal itself
    envlp_pad = np.pad(a, ((0, win_len), (0, 0)), 'constant')
    for i in range(a.shape[0]):
        envlp[i, :] = np.log(np.mean(envlp_pad[i:i+win_len, :]**2, axis=0))
    return envlp

def bfilt(data, samp_rate, fs, mode, order=3, axis = -1):
    b, a = signal.butter(order, 2*fs/samp_rate, mode)
    return signal.filtfilt(b, a, data, axis)

def compute_slp(seeg, bip, hpf=10.0, lpf=1.0, ts_on=5, ts_off=5, filter_order=5.0, remove_outliers=False):
    base_length = int(seeg['sfreq']*ts_on)

    start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
    end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq']*ts_off)
    slp = bip.get_data().T[start_idx:end_idx]

    #start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    #end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    #slp = seeg['time_series'][start_idx:end_idx]

    # Remove outliers i.e data > 2*sd
    if remove_outliers:
        for i in range(slp.shape[1]):
            ts = slp[:, i]
            ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()

    # High pass filter the data
    if hpf is not None:
        slp = bfilt(slp, seeg['sfreq'], hpf, 'highpass', axis=0)

    # Compute seeg log power
    slp = seeg_log_power(slp, 100)

    # Remove outliers i.e data > 2*sd
    if remove_outliers:
        for i in range(slp.shape[1]):
            ts = slp[:, i]
            ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()

    # Low pass filter the data to smooth
    if lpf is not None:
        slp = bfilt(slp, seeg['sfreq'], lpf, 'lowpass', axis=0)

    return slp

def replace_part_of_signal_previous(bip, seeg_info, replace_onset, replace_offset, ch_names='all'):

    start_idx = int(replace_onset * seeg_info["sfreq"])
    end_idx = int(replace_offset * seeg_info["sfreq"])
    data = bip.get_data()
    start_idx_raw = start_idx-(end_idx-start_idx)
    end_idx_raw = start_idx
    if ch_names == 'all':
        data[:,start_idx:end_idx] = data[:,start_idx_raw:end_idx_raw]
    else:
        ch_idx = [bip.ch_names.index(n) for n in ch_names]
        data[ch_idx, start_idx:end_idx] = data[ch_idx,start_idx_raw:end_idx_raw]

    info = mne.create_info(ch_names=bip.info["ch_names"], sfreq=bip.info["sfreq"])
    bip = mne.io.RawArray(data,info)

    return bip