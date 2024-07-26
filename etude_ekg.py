import matplotlib.pyplot as plt
import numpy as np
import physio 
import scipy
from tools import *
import mne
import pandas as pd
from pandasgui import show


crise=46110 
pre_ictal=45810 #5min avant crise 
post_ictal=46110+62 #fin crise 
end_time = post_ictal +1200


fichier_edf = "240407_G48A_461_459.EDF"
raw = mne.io.read_raw_edf(fichier_edf,include='1b_EKG')
srate=raw.info["sfreq"]
chanel_name=raw.info["ch_names"]
print (srate)
print(chanel_name)
time=raw.times

ekg = raw.pick_channels(['1b_EKG']).get_data()[0]
ecg, ecg_peaks = physio.compute_ecg(ekg, srate, parameter_preset='rat_ecg')
r_peak_ind = ecg_peaks['peak_index'].values


fig,ax= plt.subplots()
ax.plot(time, ecg)
ax.scatter(time[r_peak_ind], ecg[r_peak_ind], marker='o', color='magenta')
ax.set_ylabel('ekg')
plt.show()

instantaneous_rate = physio.compute_instantaneous_rate(
    ecg_peaks,
    time,
    limits=None,
    units='bpm',
    interpolation_kind='linear',
)

#smooth_rate=iirfilt(instantaneous_rate,srate=srate,highcut=0.05) #pour lisser plus on diminue la valeur
fig, ax = plt.subplots(nrows=1, sharex=True)

ax.plot(time, instantaneous_rate)
ax.set_ylabel('heart rate [bpm]')
ax.axvline(x=crise, color='r', linestyle='--') 
ax.axvline(x=post_ictal, color='r', linestyle='--') 
ax.axvline(x=end_time, color='k', linestyle='--') 
ax.set_xlabel('time [s]')

plt.show()


#histogramme 


times = np.arange(len(instantaneous_rate)) / srate
filtered_times = times[(times >= post_ictal) & (times <= end_time)]
filtered_rates = instantaneous_rate[(times >= post_ictal) & (times <= end_time)]
minutes_post_ictal = ((filtered_times - post_ictal) // 60).astype(int)

ekg_post_ictal = pd.DataFrame({
    'minute': minutes_post_ictal,
    'instantaneous_rate': filtered_rates
})
grouped_rates = ekg_post_ictal.groupby('minute')['instantaneous_rate'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(grouped_rates['minute'], grouped_rates['instantaneous_rate'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.xlabel('Minutes post-ictale')
plt.ylabel('Instantaneous Rate Moyen (unités)')
plt.title('Évolution du taux instantané moyen minute par minute (20 minutes post-ictale)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_rates['minute'])
plt.show()