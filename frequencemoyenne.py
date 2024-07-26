import matplotlib.pyplot as plt
import numpy as np
import physio 
import scipy
from tools import *
import mne
import pandas as pd
from pandasgui import show


fichier_edf = "240407_G48A_461_459.EDF"
raw = mne.io.read_raw_edf(fichier_edf, include=['1b_temp'])

srate = raw.info["sfreq"]
chanel_name = raw.info["ch_names"]
print(srate)
print(chanel_name)

time = raw.times
respi = raw.pick_channels(['1b_temp']).get_data()[0]
respi = iirfilt(respi, srate, lowcut = 1, highcut = 30)



respi = respi[47010*100:52200*100]
time= time[47010*100:52200*100]

params = physio.get_respiration_parameters('rat_plethysmo')
params['smooth'] = None
params['cycle_clean']['low_limit_log_ratio'] = 3 #abaisser = mieux identifier les phases d'apnees 
resp, resp_cycles = physio.compute_respiration(respi, srate, parameters = params)


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values
print (respi.shape)

duree_moyenne= resp_cycles['cycle_duration'].mean()
print ("duree moyenne : ", duree_moyenne)

frequence_moyenne_1 = 60 / duree_moyenne

print("Fréquence moyenne : ", frequence_moyenne_1)

amplitude_moyenne = resp_cycles['total_amplitude'].mean()

print("Amplitude moyenne : ",amplitude_moyenne)


'''
6537 8337 42510 44310 45210 47010

points = [7437 * srate, 43410 * srate, 46110 * srate]
intervalle = 900 * srate
'''

import statistics

# Déclaration de la liste de flottants
liste = [201.82692096657374,196.49597249319092]
liste2 = [507.03722363073547,506.5367041865182,475.5019239241751,476.24890218977464]
liste3 =[0.3117565595007846, 0.3206900823886515]

# Calcul de la moyenne
mean = statistics.mean(liste)
mean2 = statistics.mean(liste2)
mean3 = statistics.mean(liste3)
# Affichage de la moyenne
print(mean)
print (mean2)
print (mean3)

'''result for mean amplitude 
1.3087275939357745

result for cycle duration 
0.3117565595007846 et 0.3206900823886515

'''

