
import numpy as np
import scipy
import matplotlib.pyplot as plt


def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = 0):

    """
    IIR-Filter of signal

    -------------------
    Inputs : 
    - sig : nd array
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = scipy.signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

def calcul_med_mad(signal):
    # Calcul de la médiane
    med = np.median(signal)
    
    # Calcul de la déviation médiane absolue
    mad = np.median(np.abs(signal - med))
    
    return med, mad

def sliding_mean(sig, nwin, mode = 'same', axis = -1):
    """
    Sliding mean
    ------
    Inputs =
    - sig : nd array
    - nwin : N samples in the sliding window
    - mode : default = 'same' = size of the output (could be 'valid' or 'full', see doc scipy.signal.fftconvolve)
    - axis : axis on which sliding mean is computed (useful only in case of >= 1 dim)
    Output =
    - smoothed_sig : signal smoothed
    """
    if sig.ndim == 1:
        kernel = np.ones(nwin) / nwin
        smoothed_sig = scipy.signal.fftconvolve(sig, kernel , mode = mode)
        return smoothed_sig
    else:
        smoothed_sig = sig.copy()
        shape = list(sig.shape)
        shape[-1] = nwin
        kernel = np.ones(shape) / nwin
        smoothed_sig[:] = scipy.signal.fftconvolve(sig, kernel , mode = mode, axes = axis)
        return smoothed_sig
    
def calculate_means(resp_cycles, post_ictal, end_time):
    resp_post_ictal = resp_cycles[(resp_cycles['inspi_time'] >= post_ictal) & (resp_cycles['inspi_time'] <= end_time)]
    resp_post_ictal['minute'] = ((resp_post_ictal['inspi_time'] - post_ictal) // 60).astype(int)
    grouped_data = resp_post_ictal.groupby('minute').mean().reset_index()
    grouped_data['respiratory_rate'] = 60 / grouped_data['cycle_duration']
    return grouped_data


def plot_graphs(grouped_data, metric, ylabel, title, avg_value):
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data['minute'], grouped_data[metric], width=0.8, align='center', color='skyblue', edgecolor='black')
    plt.axhline(y=avg_value, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {avg_value}')
    plt.xlabel('Minutes post-ictale')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(grouped_data['minute'])
    plt.legend()
    plt.show()