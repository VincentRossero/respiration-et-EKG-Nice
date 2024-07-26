import matplotlib.pyplot as plt
import numpy as np
import physio 
from scipy import signal
from tools import *
import mne
import pandas as pd
from pandasgui import show

#240410_G48A459 

duree_baseline =300
date_injection =600
crise =3311
post_ictal =3311
end_time =post_ictal +2400



# duree_baseline =300
# date_injection =660
# crise =2736
# post_ictal =2785
# end_time =post_ictal +2400


# duree_baseline =295
# date_injection =600
# crise =3291
# post_ictal =3330
# end_time =post_ictal +720

fichier_edf = "240410_G48A459_ecog_ekg_temp_FLUO.EDF"
raw = mne.io.read_raw_edf(fichier_edf, include=['Temp probe'])

srate = raw.info["sfreq"]
chanel_name = raw.info["ch_names"]
print(srate)
print(chanel_name)

time = raw.times
respi = raw.pick_channels(['Temp probe']).get_data()[0]
respi = iirfilt(respi, srate, lowcut = 0.5, highcut = 20)
respi_detrend = signal.detrend (respi)

params = physio.get_respiration_parameters('rat_plethysmo')
params['smooth'] = None
params['cycle_clean']['low_limit_log_ratio'] = 3 #abaisser = mieux identifier les phases d'apnees 
resp, resp_cycles = physio.compute_respiration(respi_detrend, srate, parameters = params)
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values



#un mask pour la baseline et un pour de la crise à la fin de l'enregistrement 
mask_baseline = (resp_cycles['inspi_time'] < date_injection) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))
mask_crise = (resp_cycles['inspi_time']< end_time) & (resp_cycles['inspi_time'] > crise)

#etude des parametres 
frequence_moyenne= resp_cycles[mask_baseline]['frequence_respi'].mean()
amplitude_moyenne=resp_cycles[mask_baseline]['total_amplitude'].mean()
volume_moyen = resp_cycles[mask_baseline]['total_volume'].mean()
duree_moyenne= resp_cycles[mask_baseline]['cycle_duration'].mean()

print (frequence_moyenne)
'''
Detection des apnées : 

3*duree moyenne d'un cycle en baseline 
Ou 3 cycle minimum consecutif avec une perte d'amplitude de 80% 

'''
#les seuils
threshold = duree_moyenne * 3
seuil_amplitude = amplitude_moyenne * 0.2 

#dataframe des apnées de duree 
apnea_cycles = resp_cycles[(resp_cycles['cycle_duration'] > threshold)]
apnea_times_sec = apnea_cycles['inspi_time'].values

show (apnea_cycles)

#dataframe des apnées d'amplitude 
mask_amplitude = resp_cycles[(resp_cycles['total_amplitude']< seuil_amplitude)]


show (mask_amplitude)

# on classe dans consecutive groups les indexs du dataframe d'amplitude les groupes d'indices consecutifs 
index_amplitude = mask_amplitude.index
index_df = pd.DataFrame(index_amplitude, columns=['index'])
index_df['group'] = (index_df['index'] != index_df['index'].shift() + 1).cumsum()
consecutive_groups = index_df.groupby('group')['index'].apply(list)

#uniiquement les groupes de au moins 3 cycles d'affilés 
faible_amplitude = consecutive_groups[consecutive_groups.apply(len) >= 2]

premiers_cycles = pd.DataFrame(columns=resp_cycles.columns)
for group in faible_amplitude:
    indices = group
    total_duration = resp_cycles.loc[indices, 'cycle_duration'].sum()
    first_index = indices[0]
    premier_cycle = resp_cycles.loc[first_index].copy()
    premier_cycle['total_duration'] = total_duration
    premiers_cycles = pd.concat([premiers_cycles, premier_cycle.to_frame().T], ignore_index=True)


show (premiers_cycles)


#affichage du signal respi + apnée normal en bleu , apnee amplitude en vert 

fig,ax= plt.subplots(nrows=1,sharex=True)
# ax.plot(time,respi)
ax.plot(time,respi_detrend,color='orange')
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
ax.axvline(x=date_injection, color='r', linestyle='--')  
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 
ax.axvline(x=crise,color='k',linestyle='--')
ax.axvline(x=post_ictal,color='k',linestyle='--')

# for inspi_time in apnea_cycles['inspi_time']:
#     ax.axvline(inspi_time, color='blue', lw=2, linestyle='--')


# for inspi_time in premiers_cycles['inspi_time']:
#     ax.axvline(inspi_time, color='green', lw=2, linestyle='--')


plt.show()




fig,axs = plt.subplots(nrows= 2,sharex= True)

ax=axs[0]
ax.plot(resp_cycles['inspi_time'],resp_cycles['frequence_respi'])
ax.set_title('Frequence respiratoire ')
ax.axhline(y=frequence_moyenne,color='k',linestyle='--')
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--')
ax.axvline(x=date_injection, color='r', linestyle='--') 
ax.axvline(x=crise, color='k', linestyle='--') 
ax.axvline(x=post_ictal, color='k', linestyle='--') 
ax.set_ylabel('frequence en nombre de cycle par minute')
ax.plot()

ax=axs[1]
ax.plot(resp_cycles['inspi_time'],resp_cycles['total_amplitude'])
ax.set_title('Amplitude respiratoire')
ax.axhline(y=amplitude_moyenne,color='k',linestyle='--')
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--')
ax.axvline(x=date_injection, color='r', linestyle='--') 
ax.axvline(x=crise, color='k', linestyle='--') 
ax.axvline(x=post_ictal, color='k', linestyle='--') 
ax.set_ylabel('Amplitude totale par cycle ')
ax.plot()


plt.show()


#etude du retour à la baseline en frequence et en amplitude
mask_post_ictal = resp_cycles[(resp_cycles['inspi_time'] >= post_ictal) & (resp_cycles['inspi_time'] <= end_time)]
mask_post_ictal['minute'] = ((mask_post_ictal['inspi_time'] - post_ictal) // 60).astype(int)
grouped_data = mask_post_ictal.groupby('minute').mean().reset_index()


#on essaye de voir comment la baseline évolue sur 5 minutes 
mask_etude_baseline = resp_cycles[(resp_cycles['inspi_time'] >= date_injection-duree_baseline) & (resp_cycles['inspi_time'] <= date_injection)]
mask_etude_baseline['minute'] = ((mask_etude_baseline['inspi_time'] - (date_injection-duree_baseline)) // 60).astype(int)
grouped_data_baseline = mask_etude_baseline.groupby('minute').mean().reset_index()




plt.figure(figsize=(10, 6))
plt.bar(grouped_data['minute'], grouped_data['frequence_respi'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=frequence_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes post-ictale')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.title('Évolution de la fréquence respiratoire minute par minute ')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minute'])
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(grouped_data['minute'], grouped_data['total_amplitude'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=amplitude_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes post-ictale')
plt.ylabel('Amplitude respiratoire (cycles par minute)')
plt.title("Évolution de l'amplitude respiratoire minute par minute ")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minute'])
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(grouped_data_baseline['minute'], grouped_data_baseline['frequence_respi'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=frequence_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes baseline')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.title('Évolution de la fréquence respiratoire minute par minute ')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data_baseline['minute'])
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(grouped_data_baseline['minute'], grouped_data_baseline['total_amplitude'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=amplitude_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes baseline')
plt.ylabel('Amplitude respiratoire (cycles par minute)')
plt.title("Évolution de l'amplitude respiratoire minute par minute ")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data_baseline['minute'])
plt.legend()
plt.show()



print(f"Fréquence respiratoire moyenne: {frequence_moyenne} respirations par minute")
print(f"Amplitude respiratoire moyenne: {amplitude_moyenne} unités d'amplitude")
print(f"Volume respiratoire moyen: {volume_moyen} unités de volume")

