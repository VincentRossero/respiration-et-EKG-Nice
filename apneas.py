import matplotlib.pyplot as plt
import numpy as np
import physio
import scipy
import mne
import pandas as pd
from pandasgui import show
from tools import *

# Charger les données EDF
fichier_edf = "240411_G48A459_ecog_ekg_temp.EDF"
raw = mne.io.read_raw_edf(fichier_edf, include='1b_temp')

# Extraire les informations nécessaires
srate = raw.info["sfreq"]
time = raw.times
respi = raw.pick_channels(['1b_temp']).get_data()[0]
respi = iirfilt(respi, srate, lowcut=1.5, highcut=20)
respi = respi[:52200*100]
time = time[:52200*100]

# Définir l'événement de crise
crise = 43410 

# Calculer les paramètres de respiration
params = physio.get_respiration_parameters('rat_plethysmo')
params['smooth'] = None
params['cycle_clean']['low_limit_log_ratio'] = 3
resp, resp_cycles = physio.compute_respiration(respi, srate, parameters=params)
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

# Masquer les cycles de respiration pendant la crise
mask_crise = (resp_cycles['inspi_time'] < crise + 1200) & (resp_cycles['inspi_time'] > crise)

# Définir le seuil pour les apnées
threshold = 0.31622332094471806 * 2
apnea_cycles = resp_cycles[(resp_cycles['cycle_duration'] > threshold)]
apnea_times_sec = apnea_cycles['inspi_time'].values

mask = (apnea_cycles['inspi_time'] >= crise) & (apnea_cycles['inspi_time'] <= crise+1200)
filtered_apnea_cycles = apnea_cycles[mask]

apnee_detecte = pd.DataFrame(filtered_apnea_cycles)
apnee_detecte['minutes_since_start'] = ((apnee_detecte['inspi_time'] - crise) // 60).astype(int)


df = pd.DataFrame(apnee_detecte)
show(df) 

grouped = apnee_detecte.groupby('minutes_since_start')
sum_cycle_duration = grouped['cycle_duration'].sum()
all_minutes = pd.Series(0, index=range(20))
sum_cycle_duration = sum_cycle_duration.add(all_minutes, fill_value=0)

# Afficher le résultat
print(sum_cycle_duration)

count_apneas = grouped.size()
count_apneas = count_apneas.add(all_minutes, fill_value=0)




fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sum_cycle_duration.index, sum_cycle_duration.values, width=0.8, align='center', edgecolor='black', color='gold', alpha=0.7)
ax.set_xlabel('Minutes Since Start')
ax.set_ylabel('Total time spend in apnea Duration')
ax.set_title('Total Cycle Duration per Minute')
ax.set_xticks(range(20))
ax.set_xticklabels(range(20), rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(count_apneas.index, count_apneas.values, width=0.8, align='center', edgecolor='black', color='orange', alpha=0.7)
ax.set_xlabel('Minutes Since Start')
ax.set_ylabel('Number of Apneas')
ax.set_title('Number of Apneas per Minute')
ax.set_xticks(range(20))
ax.set_xticklabels(range(20), rotation=0)
plt.show()


# Masquer les apnées pendant la crise
mask_apnea_crise = (apnea_times_sec < crise + 1200) & (apnea_times_sec > crise)
apnea_times_during_crise = apnea_times_sec[mask_apnea_crise]


# Afficher les données et les apnées pendant la crise
fig, ax = plt.subplots(nrows=1)
ax.plot(time, respi, color='orange')
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')

# Tracer les lignes verticales pour les apnées pendant la crise
for t in apnea_times_during_crise:
    ax.axvline(t, color='k', lw=2)

plt.show()
