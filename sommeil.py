import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import mne


fichier_edf = "240410_G48A459_ecog_ekg_temp_FLUO.EDF"
raw = mne.io.read_raw_edf(fichier_edf, include=['1b_ECoG'], preload=True)
sampling_rate = raw.info["sfreq"]

# Appliquer le filtre après avoir chargé les données
#raw.filter(1., 40., fir_design='firwin')

# Extraire le signal filtré
signal = raw.pick_channels(['1b_ECoG']).get_data()[0]

# Calculer le spectrogramme avec des paramètres ajustés
windows_size_sec = 5

nperseg = int(windows_size_sec* sampling_rate)
noverlap = None
q = 0.01

f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
mask_f = (f > 1) & (f<20)
new_f = f[mask_f]
new_Sxx = Sxx[mask_f,:]
data_plot = 10*np.log10(new_Sxx)
vmin = np.quantile(data_plot,q)
vmax=np.quantile(data_plot,1-q)

# Ajuster la plage de couleurs pour mieux visualiser les détails
plt.figure(figsize=(12, 6))
# plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', vmin = np.quantile(data_plot,q), vmax=np.quantile(data_plot,1-q))  # Ajustez vmin et vmax selon vos besoins

plt.pcolormesh(t, new_f, data_plot, vmin=vmin,vmax=vmax)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
# plt.ylim(0,20)
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.show()