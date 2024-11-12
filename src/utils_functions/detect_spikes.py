''' detect interictal spikes in the signal '''
import numpy as np
import matplotlib.pyplot as plt

def get_spikes_idx(signalData, tau=256, threshold=None):
    ''' Works best with simulated data 
        Spike criteria only positive values crossing the threshold '''
    spike = False
    better_spikes_idx = []
    if threshold is None:
        threshold = np.std(signalData)*2
    for i in range(len(signalData)):
        if signalData[i] > threshold and spike == False:
            # start of spike
            spike_start = i
            spike = True
            continue
        elif signalData[i] < threshold and spike == True:
            # end of spike, we calculate and store the peak
            spike_end = i
            peak = max(signalData[spike_start:spike_end])
            peak_idx = np.where(signalData == peak)[0]
            if peak_idx.size > 1:
                peak_idx = peak_idx[np.where((spike_start <= peak_idx) & (peak_idx<= spike_end))]
                if peak_idx.size > 1:
                    print('Careful, two or more identical peaks in same spike', peak_idx)
                    peak_idx = peak_idx[0]
            elif peak_idx.size < 1:
                print('Error peak not detected')
            better_spikes_idx.append(peak_idx)
            spike = False 

    # filter out spikes too close that they can be considered as one spike
    # tau = 256 # spikes within this limit [ms] will be considered to be one spike
    opti_spikes_idx = []
    i=0
    while i < len(better_spikes_idx)-1:
        if (better_spikes_idx[i+1] - better_spikes_idx[i]) <= tau:
            peak = max(signalData[better_spikes_idx[i]], signalData[better_spikes_idx[i+1]])
            peak_idx = np.where(signalData == peak)[0]
            if peak_idx.size > 1:
                peak_idx = peak_idx[np.where((better_spikes_idx[i] <= peak_idx) & (peak_idx<= better_spikes_idx[i+1]))]
                if peak_idx.size > 1:
                    print('Careful, two or more identical maximums ? what ?', peak_idx)
                    peak_idx = peak_idx[0]
            elif peak_idx.size < 1:
                print('Error peak not detected')

            opti_spikes_idx.append(peak_idx)
            i += 2
        else:
            opti_spikes_idx.append(better_spikes_idx[i])
            i += 1
    return opti_spikes_idx, better_spikes_idx

def get_spikes_idx_realdata(signalData, tau=256, threshold=None):
    '''
    works best with real data, since we take into account here peaks
    that are both negative or positive in value

    signalData : signal of one channel with whatever length in time
    tau : max delay in time steps between two spikes, otherwise we consider them to be one spike

    Note: Interictal spikes are brief paroxysmal discharges of <250 ms duration
    '''
    spike = False
    better_spikes_idx = []
    if threshold is None:
        threshold = np.std(signalData)*2   # here defining a spike as a signal that crosses 2x the std of that signal
    for i in range(len(signalData)):
        if abs(signalData[i]) > threshold and spike == False:
            # start of spike
            spike_start = i
            spike = True
            continue
        elif abs(signalData[i]) < threshold and spike == True:
            # end of spike, we calculate and store the peak
            spike_end = i
            peak = max(abs(signalData[spike_start:spike_end]))
            peak_idx = np.where(abs(signalData) == peak)[0]
            if peak_idx.size > 1:
                peak_idx = peak_idx[np.where((spike_start <= peak_idx) & (peak_idx<= spike_end))]
                if peak_idx.size > 1:
                    print('Careful, two or more identical peaks in same spike', peak_idx)
                    peak_idx = peak_idx[0]
            elif peak_idx.size < 1:
                print('Error peak not detected')
            better_spikes_idx.append(peak_idx)
            spike = False 

    # filter out spikes too close that they can be considered as one spike
    # tau = 256, spikes within this limit will be considered to be one spike
    opti_spikes_idx = []
    i=0
    while i < len(better_spikes_idx)-1:
        if (better_spikes_idx[i+1] - better_spikes_idx[i]) <= tau:
            peak = max(abs(signalData[better_spikes_idx[i]]), abs(signalData[better_spikes_idx[i+1]]))
            peak_idx = np.where(abs(signalData) == peak)[0]
            if peak_idx.size > 1:
                peak_idx = peak_idx[np.where((better_spikes_idx[i] <= peak_idx) & (peak_idx<= better_spikes_idx[i+1]))]
                if peak_idx.size > 1:
                    print('Careful, two or more identical maximums ? what ?', peak_idx)
                    peak_idx = peak_idx[0]
            elif peak_idx.size < 1:
                print('Error peak not detected')

            opti_spikes_idx.append(peak_idx)
            i += 2
        else:
            opti_spikes_idx.append(better_spikes_idx[i])
            i += 1
    return opti_spikes_idx, better_spikes_idx


def plot_ts_and_spikes(t, signalData, spikes_idx, x0):
    fig = plt.figure(figsize=(13,5))

    plt.plot(t, signalData)
    plt.title("Epileptors time series and spike detection")
    # plt.yticks(np.arange(len(ez)),ez, fontsize=12)
    plt.legend(x0)

    plt.hlines(np.mean(signalData), 0, len(signalData), 'r')
    plt.hlines(np.std(signalData)*2, 0, len(signalData), 'orange')
    # plt.hlines(-np.std(signalData)*2, 0, len(signalData), 'orange')

    plt.xticks(fontsize=12)
    # plt.ylim([-1,len(ez)+0.5])
    # plt.xlim([t[0],t[-1]])
    plt.tight_layout()

    for el in spikes_idx:
        plt.plot(el, 0.5, 'ro')