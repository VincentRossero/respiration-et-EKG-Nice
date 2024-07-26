import physio
import neo
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import pandas as pd
from pandasgui import show
from tools import *
import mne
from scipy import signal

fichier_edf = "240407_G48A_461_459.EDF"
raw = mne.io.read_raw_edf(fichier_edf, include=['1b_temp'])

srate = raw.info["sfreq"]
chanel_name = raw.info["ch_names"]
print(srate)
print(chanel_name)

time = raw.times
respi = raw.pick_channels(['1b_temp']).get_data()[0]
respi = respi[:int(52180*srate)]
time = time[:int(52180*srate)]
respi = iirfilt(respi, srate, lowcut = 0.5, highcut = 20)
respi = signal.detrend (respi)

params = physio.get_respiration_parameters('rat_plethysmo')
params['smooth'] = None
params['cycle_clean']['low_limit_log_ratio'] = 3 #abaisser = mieux identifier les phases d'apnees 
resp, resp_cycles = physio.compute_respiration(respi, srate, parameters = params)
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values


fig, axs = plt.subplots(nrows=2,sharex=True)
ax=axs[0]
ax.plot(time,resp)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index],resp[expi_index],marker='o', color='red')
ax=axs[1]
ax.plot(resp_cycles['inspi_time'],resp_cycles['frequence_respi'])
plt.show()

resp_cycles['minutes']=(resp_cycles['inspi_time']//1800).astype(int)
grouped_data= resp_cycles.groupby('minutes').mean().reset_index()

show (grouped_data)
highlighted_bars = [3, 23, 24]

# Créer une liste de couleurs, par défaut toutes les barres seront en 'skyblue'
colors = ['skyblue'] * len(grouped_data)

# Mettre en évidence certaines barres avec une couleur différente, par exemple 'red'
for index in highlighted_bars:
    if index < len(colors):
        colors[index] = 'red'

# Tracer le graphique
plt.figure(figsize=(10, 6))
bars = plt.bar(grouped_data['minutes'], grouped_data['frequence_respi'], width=0.8, align='center',
               color=colors, edgecolor='black')

# Ajouter les étiquettes et le titre
plt.xlabel('Évolution par tranche de 30 minutes')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minutes'])

# Ajouter une légende pour les barres rouges
red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Crise d\'épilepsie')
plt.legend(handles=[red_patch], loc='upper right')

# Afficher le graphique
plt.show()