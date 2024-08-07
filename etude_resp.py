import matplotlib.pyplot as plt
import numpy as np
import physio 
import scipy
from tools import *
import mne
import pandas as pd
from pandasgui import show


'''
CRISE 1 

crise=7437 
pre_ictal=7137 #5min avant crise 
post_ictal=7437+55 #fin crise 
end_time = post_ictal +1200

CRISE 2 

crise=43410 
pre_ictal=43110 #5min avant crise 
post_ictal=43410+31 #fin crise 
end_time = post_ictal +1200

CRISE 3
crise=46110 
pre_ictal=45810 #5min avant crise 
post_ictal=46110+62 #fin crise 
end_time = post_ictal +1200
'''
crise=7437 
pre_ictal=7137 #5min avant crise 
post_ictal=7437+55 #fin crise 
end_time = post_ictal +1200

fichier_edf = "240407_G48A_461_459.EDF"
raw = mne.io.read_raw_edf(fichier_edf,include='1b_temp')

srate=raw.info["sfreq"]
chanel_name=raw.info["ch_names"]
print (srate)
print(chanel_name)

time=raw.times

respi = raw.pick_channels(['1b_temp']).get_data()[0]
respi = iirfilt(respi, srate, lowcut = 1.5, highcut = 20)
respi = respi[:52200*100]
time= time[:52200*100]


#print(ancienne_respi.shape)
print (respi.shape)
print (time.shape)

params = physio.get_respiration_parameters('rat_plethysmo')
params['smooth'] = None
params['cycle_clean']['low_limit_log_ratio'] = 3 #abaisser = mieux identifier les phases d'apnees 
resp, resp_cycles = physio.compute_respiration(respi, srate, parameters = params)


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values
print (respi.shape)


'''
mask_baseline = (resp_cycles['inspi_time'] < 14000) & (resp_cycles['inspi_time'] > 12000)
resp_baseline =resp_cycles[mask_baseline]
automatic_treshold = resp_cycles[mask_baseline]['cycle_duration'].mean() * 3
print(automatic_treshold)


apnea_cycles_baseline = resp_cycles[(resp_cycles['cycle_duration'] > automatic_treshold)]
apnea_cycles_baseline = apnea_cycles_baseline[apnea_cycles_baseline.index.to_series().diff() != 1]
apnea_times_sec_baseline = apnea_cycles_baseline['inspi_time'].values
'''
fig,axs= plt.subplots(nrows=3,sharex=True)


ax=axs[0]
ax.plot(time,respi)
ax.plot(time,respi,color='orange')
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
ax.axvline(x=crise, color='r', linestyle='--') 
ax.axvline(x=post_ictal, color='r', linestyle='--') 

'''
for t in apnea_times_sec_baseline:
    ax.axvline(t, color = 'k', lw = 2)

'''

ax=axs[1]
ax.plot(resp_cycles['inspi_time'],60/resp_cycles['cycle_duration'])
ax.set_title('Frequence respiratoire ')
ax.axvline(x=pre_ictal, color='k', linestyle='--') 
ax.axhline(y=199.16144672988233, color='k', linestyle='--')
ax.axvline(x=crise, color='r', linestyle='--') 
ax.axvline(x=post_ictal, color='r', linestyle='--')  
ax.set_ylabel('frequence en nombre de cycle par minute')
ax.plot()

ax=axs[2]
ax.plot(resp_cycles['inspi_time'],resp_cycles['total_amplitude'])
ax.set_title('Amplitude respiratoire')
ax.axvline(x=pre_ictal, color='k', linestyle='--') 
ax.axhline(y=1.3087275939357745, color='k', linestyle='--')
ax.axvline(x=crise, color='r', linestyle='--') 
ax.axvline(x=post_ictal, color='r', linestyle='--') 
ax.set_ylabel('Amplitude totale par cycle ')
ax.plot()

plt.show()




grouped_data = calculate_means(resp_cycles, post_ictal, end_time)

print(grouped_data[['minute', 'respiratory_rate', 'total_amplitude']])

plot_graphs(grouped_data, 'respiratory_rate', 'Fréquence respiratoire (cycles par minute)', 'Évolution de la fréquence respiratoire minute par minute (20 minutes post-ictale)', 199.16144672988233)
plot_graphs(grouped_data, 'total_amplitude', 'Amplitude moyenne (unités)', 'Évolution de l\'amplitude moyenne minute par minute (20 minutes post-ictale)', 1.3087275939357745)