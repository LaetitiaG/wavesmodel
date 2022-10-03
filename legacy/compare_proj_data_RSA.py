
#############################################################################
#### compare_proj_data_RSA.py
##############################################################################
# -*- coding: utf-8 -*-
"""
Correlate model predictions (from encodingModel.py) with MEG-EEG signals.
The SSVEP signal is extracted for the frequency of interest for both the 
predicted and the measured signals.
The relative amplitude and phase shift between sensors are computed in a matrix.
These matrices for measured data are compared to equivalent matrices computed 
for simulated data (similarly to representational similarity analysis). 
Created on Mon Jul  5 13:59:31 2021

@author: laeti
"""

import os
import os.path as op
import preproc
import mne 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import scipy.stats
from math import sqrt, atan2, pi, floor,exp
from mne_connectivity import plot_sensors_connectivity
import pickle
         
##############################################################################
### Functions ###
##############################################################################

def rayleigh_test(phases):
    ph = phases.reshape(-1)
    n = len(ph)
    w = np.ones(n)
    r = np.sum(w*np.exp(1j*ph))
    r = abs(r)/np.sum(w)
    R = n*r
    z = R**2 / n
    pval = exp(sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))
    return pval

def circular_mean(phases):
    '''
    Compute the circular mean of a phase vector in radian

    Parameters
    ----------
    phases : ARRAY 
        Phases vector.

    Returns
    -------
    Circular mean in radian.

    '''
    ang = np.mean(np.exp(1j*phases))
    return atan2(ang.imag, ang.real)

def circular_corr(phases1, phases2):
    # from Topics in Circular Statistics (Jammalamadaka & Sengupta, 2001, World Scientific)
    m1 = circular_mean(phases1)
    m2 = circular_mean(phases2)

    sin1 = np.sin(phases1 - m1)
    sin2 = np.sin(phases2 - m2)
    rho = np.sum(sin1*sin2)/np.sqrt(np.sum(sin1** 2)*np.sum(sin2** 2))

    # Statistical test 
    l20 = np.sum(sin1** 2)
    l02 =  np.sum(sin2** 2)
    l22 = np.sum(sin1** 2 * sin2** 2)
    zcorr = rho * np.sqrt(l20 * l02 / l22)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(zcorr)))
 
    return rho, zcorr, p_value


##############################################################################
### Parameters ###
##############################################################################

from config import cond_waves, subjects, cond_space, resultspath, simupath, sensorspath, chansel


# Choose which epochs type you want to create
# SSVEP: 1-45Hz, -1 to 2.5s, decim = 5
# 5Hzfilt: 4-6Hz, -1 to 2.5s, decim = 5
epoch_type = "SSVEP" # SSVEP 5Hzfilt
ep_name = {'SSVEP': '-', '5Hzfilt': '_5Hzfilt-'}
ch_types = ['mag','grad','eeg']
choose_ch = [['mag',False], ['grad', False], [False,True]]
  
fstim = 5 # visual stimulus frequency

# Parameters for Morlet analysis for the SSVEP extraction
freqs = np.arange(2., 50., 0.5)
n_cycles = freqs / 2.
     
# # Use constrained boundaries to fit parameters NOT USED
# bds_beta = [0,np.inf] # only positive beta
# bds_theta = [0, 2*np.pi] # phase between 0 and 2pi
# bds_inter = [-np.inf,np.inf] # no constrain
# bounds=((bds_beta[0],bds_theta[0],bds_inter[0]),(bds_beta[1],bds_theta[1],bds_inter[1]))
           
#simu = 'temporal' # simulate temporal, standing or traveling waves

# Fit on all sensors or fit on one selected sensor
comp = 'crossed' # 'matched' 'crossed' use projection matching the empirical data condition or cross 'quad_halfV1' 'quad_leftV1': model left V1

if comp == 'quad_halfV1' or comp == 'quad_leftV1':
    cond_space = ['quad']
    
    
# time of interest
tmin_crop = 0.5

# color
darkblue = [31./256, 78./256, 121./256] # model
darkgreen = [56./256, 87./256, 35./256] # measured data

# colormap for matrix
cmap = plt.get_cmap('viridis')
cmap.set_bad(color='black')
cmap_classic = plt.get_cmap('viridis')
            
##############################################################################
### Main ###
##############################################################################
  

# Create report and directories if not existing
now = datetime.now()
report_path = os.path.join(resultspath,'compare_proj_data_RSA', now.strftime("%Y%m%d_%H%M%S"))
if not os.path.isdir(report_path):
    if not os.path.isdir(os.path.join(resultspath, 'compare_proj_data_RSA')):
        os.mkdir(os.path.join(resultspath, 'compare_proj_data_RSA'))
    os.mkdir(report_path)

zscores = np.zeros((len(subjects),len(cond_space), len(cond_waves), len(ch_types), 3))  # zscore of the correlation for each condition subject*c_space*c_waves*amp/phase  /cplx  
R2_all = np.zeros((len(subjects),len(cond_space), len(cond_waves), len(ch_types), 3)) 
pval_all = np.zeros((len(subjects),len(cond_space), len(cond_waves), len(ch_types), 3)) 
ramp = np.zeros((len(subjects),len(cond_space),len(ch_types),1)); rphase = np.zeros((len(subjects),len(cond_space),len(ch_types),1)); # correlation between trav and stim simu
matrices = {'eeg': np.zeros((len(subjects), len(cond_space), len(cond_waves),2,74,74),dtype = 'complex_'), # subj*meas/pred*chan*chan
            'mag': np.zeros((len(subjects), len(cond_space), len(cond_waves),2, 102,102),dtype = 'complex_'),   
            'grad': np.zeros((len(subjects), len(cond_space), len(cond_waves),2, 102,102),dtype = 'complex_')} # complex connectivity matrices for each channel type

for s,subject in enumerate(subjects):

    # Create a new report page
    report = mne.Report(verbose=True)
                            
    # Do the procedure for each condition
    for cs, c_space in enumerate(cond_space):
        for cw,c_waves in enumerate(cond_waves):
            
            # Read epochs
            fname = '{}_{}Field_{}{}epo.fif'.format(subject, c_space, c_waves, ep_name[epoch_type])
            epochs = mne.read_epochs(os.path.join(sensorspath, fname))
            epochs.crop(tmin=0, tmax = 2)
            epochs.pick_types(meg = True, eeg = True)
        
            Nt = len(epochs.times)
            sf = epochs.info['sfreq']
            
            # Compute evoked
            evoked = epochs.average()
            
            # Read projections (name to adapt when having the good proj)
            if comp=='matched':
                sname = subject + '_proj_' + c_waves +  'Wave_' + c_space + 'Field-ave.fif'
            elif comp=='crossed':
                sname = subject + '_proj_' + cond_waves[not(cw)] +  'Wave_' + c_space + 'Field-ave.fif'
            elif comp == 'quad_halfV1':
                sname = subject + '_proj_leftV1_stimulated{}_otherHalf{}-ave.fif'.format(c_waves, cond_waves[not(cw)])
            elif comp == 'quad_leftV1':
                sname = subject + '_proj_{}_leftV1_full-ave.fif'.format(c_waves)          
            ev_proj = mne.read_evokeds(os.path.join(simupath,sname))[0]
            evokeds = [evoked, ev_proj]
            ev_name = ['measured', 'simulated']
            
            # Extract phase and amplitude at each time point at 5Hz
            phases = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt)) # train, test proj
            ampls = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt)) 
            for e, ev in enumerate(evokeds):  
                # extract instantaneous phase and amplitude
                dat = ev.data[np.newaxis,:]
                tfr = mne.time_frequency.tfr_array_morlet(dat, sfreq=sf, freqs=freqs, n_cycles=n_cycles,output='complex')
                phase = np.angle(tfr)
                amp =  np.abs(tfr)
                
                # extract instantaneous phase and amplitude at frequency of interest
                phases[e] = np.squeeze(phase[0,:,freqs == fstim])
                ampls[e] = np.squeeze(amp[0,:,freqs == fstim])

            msk_t = evoked.times > tmin_crop            

            ###### AMPLITUDE ######
            # Sanity check: estimate the variation of instantaneous phase/amplitude through time
            ampls_m = np.mean(ampls[:,:,msk_t],2) # mean measured phases
            ampls_sd = np.std(ampls[:,:,msk_t],2)
            
            for i in range(len(evokeds)):
                f= plt.figure()
                plt.hist(ampls_sd[i]/ampls_m[i]*100)
                plt.title('Coefficient of variation (sd/mean*100) in %')
                plt.ylabel('Number of sensors')
                report.add_figs_to_section(f, captions='Variation in amplitude estimation of across time - ' + ev_name[i], section='{}Field_{}'.format(c_space,c_waves))
            
            # Plot histogram across sensors of the standard deviation for each sensor type
            f = plt.figure()
            for ch,ch_type in enumerate(ch_types):
                inds = mne.pick_types(epochs.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
                plt.subplot(1,3,ch+1)
                plt.hist(ampls_sd[0, inds]/ampls_m[0, inds]*100)
                plt.title(ch_type)
            report.add_figs_to_section(f, captions='Variation in amplitude estimation per sensor - measured', section='{}Field_{}'.format(c_space,c_waves))
                        
            # Compute a matrix of relative amplitude between sensors for each sensor type
            cov_amp = []
            for ch,ch_type in enumerate(ch_types):
                
                if ch_type == 'grad':
                    chan_plot = 'mag'
                    # do the average of relative amplitude between grad1 and grad2
                    inds1 = mne.pick_types(epochs.info, meg = 'planar1', eeg = False)
                    inds2 = mne.pick_types(epochs.info, meg = 'planar2', eeg = False)
                    ind_grad = [inds1, inds2]
                    cov_tmp = np.zeros((2,len(evokeds), len(inds), len(inds)))
                    for j in range(len(ind_grad)):
                        inds = ind_grad[j]
                        for i in range(len(evokeds)): # for measured and predicted data
                            ampls_ch = ampls[i,inds][:,msk_t]    #ampls[i,inds,msk_t]
                            # loop across rows
                            for r in range(len(inds)):
                                for c in range(len(inds)): # loop across column
                                    cov_tmp[j,i,r,c] = np.mean(ampls_ch[r]/ampls_ch[c])
                    cov_ch = cov_tmp.mean(0)
                else:
                    chan_plot = chansel[ch_type][0] 
                    
                    inds = mne.pick_types(epochs.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
                    cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
                    
                    for i in range(len(evokeds)): # for measured and predicted data
                        ampls_ch = ampls[i,inds][:,msk_t] 
                        # loop across rows
                        for r in range(len(inds)):
                            for c in range(len(inds)): # loop across column
                                cov_ch[i,r,c] = np.mean(ampls_ch[r]/ampls_ch[c]) # average across time
                
                                        
                cov_amp.append(np.log(cov_ch))
            
                # plot covariance matrix
                f = plt.figure(figsize=(12, 6))
                plt.title('log(relative amplitude)')
                for i in range(len(evokeds)):
                    plt.subplot(1,3,i+1)
                    plt.imshow(np.log(cov_ch[i]))
                    plt.colorbar()
                    plt.title(ev_name[i])
        
                diff = np.log(cov_ch[0])-np.log(cov_ch[1])
                threshold = diff/(np.log(cov_ch[0])+np.log(cov_ch[1]))
                thresh = 0.01
                msk_black = (diff<thresh) & (diff>-thresh)
                np.ma.masked_where(msk_black, diff)
                diff = np.ma.masked_where(msk_black, diff)
                plt.subplot(1,3,3) # plot the diff
                plt.imshow(diff, cmap =cmap)        
                plt.colorbar()
                plt.title('measured-simulated')        
                report.add_figs_to_section(f, captions='covariance amplitude - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
    
                # Correlate measured vs predicted matrix
                msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
                meas = cov_ch[0][msk_tri]
                simu = cov_ch[1][msk_tri]
                R2, pval = scipy.stats.spearmanr(np.log(meas), np.log(simu))
                zscores[s, cs, cw, ch, 0] = 0.5*np.log((1+R2)/(1-R2)) # fisher Z (ok for spearman when N>10)
                R2_all[s, cs, cw, ch, 0] = R2
                pval_all[s, cs, cw, ch, 0] = pval
                
                # plot the correlation
                f=plt.figure()
                # plt.plot(np.log(np.ndarray.flatten(cov_ch[0])), np.log(np.ndarray.flatten(cov_ch[1])), 'o', markersize=2)
                plt.plot(np.log(meas), np.log(simu), 'o', markersize=2)
                plt.xlabel('log(relative amplitude) measured')
                plt.ylabel('log(relative amplitude) simulated')
                plt.title('R2={:.2f}, p={:.3f}'.format(R2, pval))
                report.add_figs_to_section(f, captions='correlation amplitude - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
                
                # Find which sensors correlates the better: how to do that?
                np.triu(msk_black, k = 1) # zero the diagonal and all values below
                channel_names = np.array(evoked.ch_names)[inds]
                chan_corr = channel_names[np.any(np.triu(msk_black, k = 1),0)]
                
                # Plot topo with these channels
                template = evoked.copy().crop(tmin=0, tmax=0.002)
                template.data = np.zeros((np.shape(template.data)))
                ind_ch= mne.pick_channels(evoked.ch_names, chan_corr)
                mask = np.zeros((len(np.array(evoked.ch_names)), 1), dtype='bool') 
                mask[ind_ch] = True
                f2 = template.plot_topomap(times = 0, ch_type= ch_type, vmax = 1000,mask = mask, 
                              mask_params=dict(markerfacecolor='k', markersize=8), colorbar=False,
                              time_format ='',title=ch_type)
                report.add_figs_to_section(f2, captions='Channels showing relative amplitude similar to simulated data - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
                                  
                # Plot correlation for these channels
                f=plt.figure()
                plt.plot(np.log(cov_ch[0][msk_black & msk_tri]), np.log(cov_ch[1][msk_black & msk_tri]), 'o', markersize=2)
                plt.xlabel('log(relative amplitude) measured')
                plt.ylabel('log(relative amplitude) simulated')
                plt.title('R2={:.2f}, p={:.3f}'.format(R2, pval))
                report.add_figs_to_section(f, captions='correlation amplitude (best chans) - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))            
                
                # Plot connectivity plot           
                f = mne.viz.plot_connectivity_circle(1/np.abs(diff), channel_names, vmin=0 , vmax = 1/0.01,
                                                 title = '1/(amplitude meas-simu)')
                report.add_figs_to_section(f[0], captions='connectivity amplitude - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))            
    
                # only the connection around 0 are plotted
                diff = np.log(cov_ch[0]) - np.log(cov_ch[1])
                np.fill_diagonal(diff,1)
                f = plot_sensors_connectivity(template.copy().pick_types(meg= chan_plot, eeg=chansel[ch_type][1]).info, diff, thresh = thresh)
                report.add_figs_to_section(f, captions='Channel pairs with similar ampl +/-' + str(thresh) + ' - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
                
            ###### PHASE ######
            # Compute a matrix of relative phase between sensors for each sensor type
            cov_phase = []
            for ch,ch_type in enumerate(ch_types):  
                if ch_type == 'grad':
                    chan_plot = 'mag'
                    # do the average of relative amplitude between grad1 and grad2
                    inds1 = mne.pick_types(epochs.info, meg = 'planar1', eeg = False)
                    inds2 = mne.pick_types(epochs.info, meg = 'planar2', eeg = False)
                    inds = inds2
                    cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
                    for i in range(len(evokeds)): # for measured and predicted data
                        phases_1 = phases[i,inds1][:,msk_t]  
                        phases_2 = phases[i,inds2][:,msk_t] 
                        # loop across rows
                        for r in range(len(inds)):
                            for c in range(len(inds)): # loop across column
                                d1 = phases_1[r] - phases_1[c]
                                d11 = (d1 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi
                                
                                d2 = phases_2[r] - phases_2[c]
                                d22 = (d2 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi   
                                
                                cov_ch[i,r,c] = circular_mean(np.array([circular_mean(d11), circular_mean(d22)]))
    
                else:    
                    chan_plot = chansel[ch_type][0]  
                    
                    inds = mne.pick_types(epochs.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
                    cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
                
                    for i in range(len(evokeds)): # for measured and predicted data
                        phases_ch = phases[i,inds][:,msk_t]
                        # loop across rows
                        for r in range(len(inds)):
                            for c in range(len(inds)): # loop across column
                                d1 = phases_ch[r] - phases_ch[c]
                                d = (d1 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi
                                cov_ch[i,r,c] = circular_mean(d)
                                        
                cov_phase.append(cov_ch)
                
                # plot covariance matrix
                f = plt.figure(figsize=(12, 6))
                plt.title('phase shift')
                for i in range(len(evokeds)):
                    plt.subplot(1,3,i+1)
                    plt.imshow(cov_ch[i], cmap='twilight', interpolation='nearest')
                    plt.colorbar()
                    plt.title(ev_name[i])
        
                diff = cov_ch[0]-cov_ch[1]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi
                msk_black = (diff<thresh) & (diff>-thresh) # 0.01 rad = 0.57°
                np.ma.masked_where(msk_black, diff)
                diff = np.ma.masked_where(msk_black, diff)
                cmap = plt.get_cmap('twilight')
                cmap.set_bad(color='black')
                plt.subplot(1,3,3) # plot the diff
                plt.imshow(diff, cmap =cmap, interpolation='nearest')        
                plt.colorbar()
                plt.title('measured-simulated')        
                report.add_figs_to_section(f, captions='covariance phase - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
                
                # Correlate measured vs predicted matrix
                msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
                meas = cov_ch[0][msk_tri]
                simu = cov_ch[1][msk_tri]                    
                R2, zscore, pval = circular_corr(meas, simu)
                zscores[s, cs, cw, ch, 1] = zscore
                R2_all[s, cs, cw, ch, 1] = R2
                pval_all[s, cs, cw, ch, 1] = pval
                
                mean_diff = circular_mean(meas-simu)  
                  
                # plot the circular correlation
                f = plt.figure()
                ax = f.add_subplot(1,2,1, polar=True)
                plt.title('b: meas, r: simu')
                plt.suptitle('R2={:.2f}, p={:.3f}'.format(R2, pval))
                plt.hist(meas,bins=20,range=(-pi,pi),alpha=0.3,color='b')
                plt.hist(simu,bins=20,range=(-pi,pi),alpha=0.4,color='r')
                ax = f.add_subplot(1,2,2, polar=True)
                plt.title('measured-simulated')
                plt.hist(meas-simu,bins=20,range=(-pi,pi),alpha=0.3,color='k') 
                plt.plot(np.array([mean_diff,mean_diff]),np.array([0,500]),'--k', linewidth=3)
                #plt.xlabel(str(freq))
                plt.xticks([])
                plt.yticks([])
                report.add_figs_to_section(f, captions='correlation phase - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
    
                # # copy the data into csv to import in R
                # fname = 'C:\\Users\\laeti\\Data\\wave_model\\testR\\data2XX_measuredPhase.csv'
                # np.savetxt(fname, meas, delimiter=",")
                # fname = 'C:\\Users\\laeti\\Data\\wave_model\\testR\\data2XX_simuPhase.csv'
                # np.savetxt(fname, simu, delimiter=",")
                
                # Find which sensors correlates the better: how to do that?
                channel_names = np.array(evoked.ch_names)[inds]
                chan_corr = channel_names[np.any(np.triu(msk_black, k = 1),0)] # zero the diagonal and all values below
                
                # Plot topo with these channels
                template = evoked.copy().crop(tmin=0, tmax=0.002)
                template.data = np.zeros((np.shape(template.data)))
                ind_ch= mne.pick_channels(evoked.ch_names, chan_corr)
                mask = np.zeros((len(np.array(evoked.ch_names)), 1), dtype='bool') 
                mask[ind_ch] = True
                f2 = template.plot_topomap(times = 0, ch_type= ch_type, vmax = 1000,mask = mask, 
                              mask_params=dict(markerfacecolor='k', markersize=8), colorbar=False,
                              time_format ='',title=ch_type)
                report.add_figs_to_section(f2, captions='Channels showing relative phase similar to simulated data - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
            
                # only the connection around 0 are plotted
                diff = abs(np.log(cov_ch[0]) - np.log(cov_ch[1]))
                np.fill_diagonal(diff,1)
                diff[np.isnan(diff)] = 1
                if np.sum(diff <thresh) !=0:
                    f = plot_sensors_connectivity(template.copy().pick_types(meg= chan_plot, eeg=chansel[ch_type][1]).info, diff, thresh = thresh)
                    report.add_figs_to_section(f, captions='Channel pairs with similar phase +/-' + str(thresh) + ' - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
    
            ###### CPLEX NBER ######
            cplx = ampls*np.exp(phases*1j)
            
            # Compute a complex matrix combinining amplitude and phase
            cov_clx = []
            for ch,ch_type in enumerate(ch_types): 
                if ch_type == 'grad':
                    inds = mne.pick_types(epochs.info, meg = 'planar1', eeg = choose_ch[ch][1])
                    chan_plot = 'mag'
                else:
                    inds = mne.pick_types(epochs.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
                    chan_plot = choose_ch[ch][0]
                    
                # Complex matrix
                covX = cov_amp[ch]*np.exp(cov_phase[ch]*1j)
                cov_clx.append(covX)
            
                matrices[ch_type][s,cs, cw,:,:, :] = covX
                
                # Correlate matrix using conjugate
                msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
                meas = covX[0][msk_tri]
                simu = covX[1][msk_tri]        
                
                # amplitude of the complex correlation coefficient
                R2 = np.abs(np.corrcoef(meas,simu))[1,0]
                zscores[s, cs, cw, ch, 2] = 0.5*np.log((1+R2)/(1-R2)) # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
                df = len(meas)-2
                tscore = R2*np.sqrt(df)/np.sqrt(1-R2*R2) # Kanji 2006
                pval = scipy.stats.t.sf(tscore, df)
                pval_all[s, cs, cw, ch, 2] = pval
                R2_all[s, cs, cw, ch, 2] = R2
                
                            
            ###### AMPLITUDE & PHASE ######
            # Search for channels having both a phase and amplitude close to simu       
            for ch,ch_type in enumerate(ch_types): 
                if ch_type == 'grad':
                    inds = mne.pick_types(epochs.info, meg = 'planar1', eeg = choose_ch[ch][1])
                    chan_plot = 'mag'
                else:
                    inds = mne.pick_types(epochs.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
                    chan_plot = choose_ch[ch][0]
                diff_amp = cov_amp[ch][0]-cov_amp[ch][1]
                diff_phase = cov_phase[ch][0]-cov_phase[ch][1]
                thresh = 0.05
                msk_phase = (diff_phase<thresh) & (diff_phase>-thresh)
                msk_amp = (diff_amp<thresh) & (diff_amp>-thresh)
                msk_both = msk_phase & msk_amp
                
                # Find which sensors correlates the better: how to do that?
                channel_names = np.array(evoked.ch_names)[inds]
                chan_corr = channel_names[np.any(np.triu(msk_both, k = 1),0)] # zero the diagonal and all values below
                
                # Plot topo with these channels
                template = evoked.copy().crop(tmin=0, tmax=0.002)
                template.data = np.zeros((np.shape(template.data)))
                ind_ch= mne.pick_channels(evoked.ch_names, chan_corr)
                mask = np.zeros((len(np.array(evoked.ch_names)), 1), dtype='bool') 
                mask[ind_ch] = True
                f2 = template.plot_topomap(times = 0, ch_type= ch_type, vmax = 1000,mask = mask, 
                              mask_params=dict(markerfacecolor='k', markersize=8), colorbar=False,
                              time_format ='',title=ch_type)
                report.add_figs_to_section(f2, captions='Channels with similar phase & ampli +/-0.05 - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
                             
                # only the connection around 0 are plotted 
                mcon = np.ones(np.shape(cov_amp[ch][0]))
                mcon[msk_both] = 0.001
                np.fill_diagonal(mcon,1)
                if np.sum(mcon <thresh) !=0:
                    f = plot_sensors_connectivity(template.copy().pick_types(meg= chan_plot, eeg=chansel[ch_type][1]).info, mcon, thresh = thresh)
                    report.add_figs_to_section(f, captions='Channel pairs with similar phase & ampli +/-' + str(thresh) + ' - ' + ch_type, section='{}Field_{}'.format(c_space,c_waves))
    
     
        # Look at how much model predictions correlates together (trav and stand)
        for ch,ch_type in enumerate(ch_types): 
            inds = len(cov_amp[ch][0])
            msk_tri = np.triu(np.ones((inds, inds), bool), k = 1) 
            r, p = scipy.stats.spearmanr(cov_amp[ch][0][msk_tri], cov_amp[ch][1][msk_tri])
            ramp[s,cs,ch] = r
            r, z, p = circular_corr(cov_phase[ch][0][msk_tri], cov_phase[ch][1][msk_tri])
            rphase[s,cs,ch] = r
        
    # # Save zscore of the  correlation coefficient
    # z1 = zscores_matched # space(full/quad)*waves(trav/stand)*chtype(mag/grad/eeg)*ampl/phase
    # z2 = zscores_crossed
    # N = np.zeros((len(cond_space), len(cond_waves), len(ch_types), 2))
    # N[:,:,0:2] = 102 # mag & grad
    # N[:,:,2] = 74 #J eeg
    # Zobserved = (z1 - z2) / np.sqrt(1.06/(N - 3) + 1.06/(N-3))
    
    # p_values = 2 * (1 - scipy.stats.norm.cdf(abs(Zobserved)))
               
    # Save report
    report.save(os.path.join(report_path, 'report_' + subject + '_comp' + comp + '.html'), overwrite=True)    

# Save R2 values for all subjects
RSA_results = {"zscores": zscores,
               "R2": R2_all,
               "pval":pval_all,
               "R2_simu_ampl": ramp,
               "R2_simu_phase": rphase,
               "subjects": subjects,
               "cplx_mat": matrices
               }
fname = "RSA_results_allSubj_comp" + comp + ".pkl"
a_file = open(os.path.join(simupath, fname), "wb")
pickle.dump(RSA_results, a_file)
a_file.close()

