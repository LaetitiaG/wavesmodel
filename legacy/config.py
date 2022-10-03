#############################################################################
## wave_model - configuration file ##
#############################################################################

# Paths
wdir = '../data/'
datapath = wdir + 'data_MEEG\\raw\\'
preprocpath = wdir + 'data_MEEG\\preproc\\'
sensorspath = wdir + 'data_MEEG\\sensors\\' 
simupath = wdir + 'data_MEEG\\simulation\\' 
resultspath = wdir + 'results\\'
subjects_dir = wdir + 'data_MRI\\preproc\\freesurfer\\'

# for EEG from INCC
datapath_EEG = wdir + 'data_EEG_INCC\\raw\\'
datapath_ET = wdir + 'data_EEG_INCC\\behav_ET\\'
preprocpath_EEG = wdir + 'data_EEG_INCC\\preproc\\'
sensorspath_EEG = wdir + 'data_EEG_INCC\\sensors\\' 

# Participants
#subjects = ['P01']
subjects = ['2XXX72', 'W80R0V', '7TEO5P', 'FADM9A', 'RGTMSZ', '5P57E6', 'UNI5M7', 'D1K7TN', 'H9W70R', 'OF4IP5', 'XZJ7KI', 'S1VW75', 'JRRWDT']
subjects_INCC = ['S01']

# INCC (n blocks)
Nruns = {'S01':3}

###########################################################
################### experimental design ###################
###########################################################

cond_space = ['full', 'quad']
cond_waves = ['trav', 'stand']
cond_ses2 = ['trav_out','stand', 'fov_out','trav_in']
col_cond = {'trav': 'crimson', 'stand':'dodgerblue', 'trav_out': 'crimson', 'trav_in': 'darkorange', 'fov_out': 'darkmagenta'}
line_cond = {'full': '-', 'quad': '--'}


###########################################################
################### channel selection ###################
###########################################################

chanType = ['mag','grad','eeg']
chansel = {'mag': ['mag', False],'grad': ['grad', False],'eeg': [False, True]}

chan_occipital = {'mag': ['MEG1711', 'MEG1731', 'MEG1941', 'MEG1911', 'MEG1741', 'MEG2041',
             'MEG1921', 'MEG1931', 'MEG2141', 'MEG2031', 'MEG2111', 'MEG2121', 
             'MEG2131', 'MEG2331', 'MEG2341', 'MEG2311', 'MEG2321', 'MEG2511', 
             'MEG2531', 'MEG2541'],
                  'grad':['MEG1712', 'MEG1732', 'MEG1942', 'MEG1912', 'MEG1742', 'MEG2042',
             'MEG1922', 'MEG1932', 'MEG2142', 'MEG2032', 'MEG2112', 'MEG2122', 
             'MEG2132', 'MEG2332', 'MEG2342', 'MEG2312', 'MEG2322', 'MEG2512', 
             'MEG2532', 'MEG2542', 'MEG1713', 'MEG1733', 'MEG1943', 'MEG1913', 'MEG1743', 'MEG2043',
             'MEG1923', 'MEG1933', 'MEG2143', 'MEG2033', 'MEG2113', 'MEG2123', 
             'MEG2133', 'MEG2333', 'MEG2343', 'MEG2313', 'MEG2323', 'MEG2513', 
             'MEG2533', 'MEG2543'],
                  'eeg': ['EEG062', 'EEG063','EEG064','EEG065','EEG066','EEG067','EEG068',
                          'EEG069', 'EEG070', 'EEG071','EEG072','EEG073', 'EEG074']}


###########################################################
######################### PREPROC #########################
###########################################################

# tsss
st_correlation = 0.980

# Delay measured between trigger and physical stimuli timing
delay_trig = 0.023

# bad channels (manually detected, in addition to the automatically detected)
badC_MEG = {'P01': 
             {'block01_2runs.fif': [], 
              'block02.fif': ['MEG0643'], 
              'block03.fif': ['MEG0643'],
              'block04.fif': [],
              'block05.fif': ['MEG0643'],
              'block06.fif': ['MEG0643'],
              'block07.fif': ['MEG0643'],
              'block08.fif': ['MEG0643'],
              'block09.fif': ['MEG0643', 'MEG2322'],
              'rest_YF.fif': ['MEG0643'],
              'rest_YF_2.fif': ['MEG0643'],
              'rest_YO.fif': ['MEG0643'],
              'rest_YO_2.fif': ['MEG0643'],
              'empty_room.fif': []},
             '2XXX72': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142', 'MEG1243'], 
              'run02.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'], 
              'run03.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run04.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run05.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run06.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run07.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run08.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'run09.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142', 'MEG0913'],
              'run10.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG2143', 'MEG0142'],
              'resting_stateYF.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'resting_stateYF_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'resting_stateYO.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142'],
              'resting_stateYO_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142', 'MEG0913'],
              'empty_room.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0142']},    
             'W80R0V': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1022', 'MEG1333', 'MEG2423'], 
              'run02.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0943'], 
              'run03.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0943','MEG2623'],
              'run04.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0943'],
              'run05.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0943', 'MEG0622','MEG2423'],
              'run06.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'run07.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132', 'MEG1521'],
              'run08.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'run09.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'run10.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'resting_stateYF.fif': ['MEG0341', 'MEG1433', 'MEG2322'],
              'resting_stateYF_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'resting_stateYO.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'resting_stateYO_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132'],
              'empty_room.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132', 'MEG0133', 'MEG1022', 'MEG1243']},
             '7TEO5P': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'], 
              'run02.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'], 
              'run03.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run04.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run05.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run06.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run07.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run08.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run09.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'run10.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'resting_stateYF.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'resting_stateYF_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'resting_stateYO.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'resting_stateYO_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312'],
              'empty_room.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1312']},
             'FADM9A': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1041', 'MEG1412', 'MEG1022', 'MEG0532', 'MEG0132', 'MEG1411'],
              'run02.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0532'], 
              'run03.fif': ['MEG0132','MEG0341', 'MEG1433', 'MEG2322', 'MEG0532'],
              'run04.fif': ['MEG0132', 'MEG0341', 'MEG1022', 'MEG1413', 'MEG1433', 'MEG2322', 'MEG0532'],
              'run05.fif': ['MEG0132', 'MEG0341', 'MEG1243', 'MEG1433', 'MEG1531', 'MEG2322', 'MEG0532'],
              'run06.fif': ['MEG0332', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1243', 'MEG1333', 'MEG1433', 'MEG2322'],
              'run07.fif': ['MEG0341', 'MEG0532', 'MEG1022', 'MEG1433', 'MEG2021', 'MEG2143', 'MEG2322', 'MEG2623'],
              'run08.fif': ['MEG0132', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1333', 'MEG1433', 'MEG2322'],
              'run09.fif': ['MEG0132','MEG0332','MEG0341', 'MEG0532', 'MEG1022', 'MEG1243', 'MEG1433', 'MEG2011', 'MEG2021', 'MEG2322', 'MEG2623'],
              'run10.fif': ['MEG0341', 'MEG0532', 'MEG1433', 'MEG2011', 'MEG2322'],
              'resting_stateYF.fif': ['MEG0132', 'MEG0332', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1243', 'MEG1333', 'MEG1412', 'MEG1433', 'MEG2031', 'MEG2322', 'MEG2623'],
              'resting_stateYF_2.fif': ['MEG0332', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1243', 'MEG1433', 'MEG2011', 'MEG2322'],
              'resting_stateYO.fif': ['MEG0132', 'MEG0332', 'MEG0532', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1041', 'MEG1433', 'MEG2322'],
              'resting_stateYO_2.fif': ['MEG0132', 'MEG0332', 'MEG0341', 'MEG0532', 'MEG0921', 'MEG1022', 'MEG1433', 'MEG2021', 'MEG2322'],
              'empty_room.fif': ['MEG0132', 'MEG0332', 'MEG0341', 'MEG0532', 'MEG1022', 'MEG1243', 'MEG1433', 'MEG2322']} ,
             'RGTMSZ': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913', 'MEG1333','MEG2223', 'MEG0521'], #0341,1433,2322 are always bad
              'run02.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913', 'MEG1022', 'MEG1333', 'MEG0521'], 
              'run03.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913','MEG1333'],
              'run04.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913','MEG1333'],
              'run05.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913','MEG1333'],
              'run06.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913','MEG1333'],
              'run07.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913','MEG1333'],
              'run08.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0521'],
              'run09.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0913', 'MEG0521'],
              'run10.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0521'],
              'resting_stateYF.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1333'],
              'resting_stateYF_2.fif': ['MEG0341', 'MEG1433', 'MEG2322',  'MEG0521', 'MEG0642'],
              'resting_stateYO.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG1333'],
              'resting_stateYO_2.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0521'],
              'empty_room.fif': ['MEG0341', 'MEG1433', 'MEG2322']},    
             '5P57E6': 
             {'run01.fif': ['MEG0341', 'MEG0913', 'MEG1433', 'MEG2322'], 
              'run02.fif': ['MEG0341', 'MEG0913', 'MEG1433', 'MEG2322', 'MEG0413', 'MEG1022'], 
              'run03.fif': ['MEG0341', 'MEG0913', 'MEG1433', 'MEG2322', 'MEG1243', 'MEG1022'],
              'run04.fif': ['MEG0341', 'MEG0132', 'MEG1433', 'MEG2322', 'MEG0413', 'MEG1022', 'MEG1132'],
              'run05.fif': ['MEG0341', 'MEG0132', 'MEG1433', 'MEG2322','MEG2122', 'MEG1132'],
              'run06.fif': ['MEG0341', 'MEG0132', 'MEG1433', 'MEG2322', 'MEG0322', 'MEG1022', 'MEG1132', 'MEG2122'],
              'run07.fif': ['MEG0341', 'MEG0132', 'MEG1433', 'MEG2322','MEG0322', 'MEG1132'],
              'run08.fif': ['MEG0341', 'MEG0132', 'MEG1132', 'MEG1022', 'MEG1433', 'MEG2322'],
              'run09.fif': ['MEG0132', 'MEG0412', 'MEG1132', 'MEG1243', 'MEG1433', 'MEG2322','MEG0341'],
              'run10.fif': ['MEG0132', 'MEG0412', 'MEG1132', 'MEG1022', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYF.fif': ['MEG0132', 'MEG1022', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYF_2.fif': ['MEG0132', 'MEG0412', 'MEG1132', 'MEG1022', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYO.fif': ['MEG0132','MEG0913', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYO_2.fif': ['MEG0132', 'MEG0412','MEG1132','MEG1243','MEG1433', 'MEG2322','MEG0341'],
              'empty_room.fif': []},  
            'D1K7TN': 
             {'run01.fif': ['MEG0322', 'MEG0342', 'MEG1433', 'MEG2322', 'MEG2623', 'MEG0341', 'MEG0521', 'MEG0222'], 
              'run02.fif': ['MEG0322', 'MEG1022', 'MEG2322', 'MEG2212',  'MEG0341'], 
              'run03.fif': ['MEG1022', 'MEG1433', 'MEG0341'],
              'run04.fif': ['MEG0132', 'MEG0322', 'MEG1022', 'MEG1433', 'MEG2322', 'MEG0341'],
              'run05.fif': ['MEG0322', 'MEG0723', 'MEG1433', 'MEG2322','MEG0341'],
              'run06.fif': ['MEG0723', 'MEG1433', 'MEG2322', 'MEG0341'],
              'run07.fif': ['MEG0322', 'MEG0323', 'MEG1433', 'MEG2322','MEG0341'],
              'run08.fif': ['MEG0322', 'MEG1433', 'MEG2322', 'MEG0341'],
              'run09.fif': ['MEG0323', 'MEG1433', 'MEG2322', 'MEG2623','MEG0341'],
              'run10.fif': ['MEG1022', 'MEG1433', 'MEG1833', 'MEG2322','MEG0341'],
              'resting_stateYF.fif': ['MEG0322', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYF_2.fif': ['MEG0322', 'MEG1022', 'MEG1433','MEG2322', 'MEG0341'],
              'resting_stateYO.fif': ['MEG0322', 'MEG1433', 'MEG2322','MEG0341'],
              'resting_stateYO_2.fif': ['MEG1022', 'MEG1433', 'MEG2322','MEG0341'],
              'empty_room.fif': []},              
             'UNI5M7': 
             {'run01.fif': ['MEG0341', 'MEG1433', 'MEG2322', 'MEG0132', 'MEG0723', 'MEG1043', 'MEG1243', 'MEG2312'], 
              'run02.fif': ['MEG0132','MEG1022','MEG1043','MEG1433','MEG1243','MEG2322','MEG0341'], 
              'run03.fif': ['MEG0132','MEG1022','MEG1043','MEG1433','MEG1243','MEG2322','MEG0341'],
              'run04.fif': ['MEG0132','MEG1022','MEG1043','MEG1433','MEG1243','MEG2322','MEG0341'],
              'run05.fif': ['MEG0132','MEG0322','MEG1022','MEG1043','MEG1433','MEG1243','MEG2322','MEG2122','MEG0341'],
              'run06.fif': ['MEG0132','MEG0322','MEG1043','MEG1433','MEG2322','MEG2122','MEG0341'],
              'run07.fif': ['MEG0132','MEG1043','MEG1022','MEG1433','MEG2322','MEG0341'],
              'run08.fif': ['MEG0132','MEG1043','MEG1022','MEG1433','MEG2322','MEG0341'],
              'run09.fif': ['MEG0132','MEG1043','MEG1022','MEG1433','MEG2122','MEG2322','MEG0341'],
              'run10.fif': ['MEG0132','MEG1043','MEG1433','MEG2322','MEG0341'],
              'resting_stateYF.fif': ['MEG0132','MEG0322','MEG1043','MEG1433','MEG2322','MEG0341'],
              'resting_stateYF_2.fif': ['MEG0132','MEG0322','MEG1043','MEG1022','MEG1243', 'MEG1433','MEG2322','MEG0341'],
              'resting_stateYO.fif': ['MEG0132','MEG1043','MEG1243','MEG1433','MEG2322','MEG0341', 'MEG2312'],
              'resting_stateYO_2.fif': ['MEG0132','MEG1043','MEG1243','MEG1433','MEG2322','MEG0341', 'MEG2312'],
              'empty_room.fif': []},               
             'XX': 
             {'run01.fif': [], 
              'run02.fif': [], 
              'run03.fif': [],
              'run04.fif': [],
              'run05.fif': [],
              'run06.fif': [],
              'run07.fif': [],
              'run08.fif': [],
              'run09.fif': [],
              'run10.fif': [],
              'resting_stateYF.fif': [],
              'resting_stateYF_2.fif': [],
              'resting_stateYO.fif': [],
              'resting_stateYO_2.fif': [],
              'empty_room.fif': []}             
             }     
    
badC_EEG = {'P01': 
         {'block01_2runs.fif': ['EEG029', 'EEG039'], 
          'block02.fif': ['EEG029', 'EEG039'], 
          'block03.fif': ['EEG029', 'EEG039'],
          'block04.fif': ['EEG029', 'EEG039'],
          'block05.fif': ['EEG029', 'EEG039'],
          'block06.fif': ['EEG029', 'EEG039'],
          'block07.fif': ['EEG029', 'EEG039'],
          'block08.fif': ['EEG029', 'EEG039'],
          'block09.fif': ['EEG029', 'EEG039'],
          'rest_YF.fif': ['EEG029', 'EEG039'],
          'rest_YF_2.fif': ['EEG029', 'EEG039'],
          'rest_YO.fif': ['EEG029', 'EEG039'],
          'rest_YO_2.fif': ['EEG029', 'EEG039']},
         '2XXX72': 
             {'run01.fif': ['EEG027', 'EEG039', 'EEG062', 'EEG050'], 
              'run02.fif': ['EEG028', 'EEG039', 'EEG050'], 
              'run03.fif': ['EEG028', 'EEG039', 'EEG050'],
              'run04.fif': ['EEG027', 'EEG039', 'EEG050'],
              'run05.fif': ['EEG027', 'EEG039', 'EEG050'],
              'run06.fif': ['EEG027', 'EEG039', 'EEG050'],
              'run07.fif': ['EEG027', 'EEG039', 'EEG050'],
              'run08.fif': ['EEG028', 'EEG039', 'EEG050'],
              'run09.fif': ['EEG027', 'EEG039', 'EEG050'],
              'run10.fif': ['EEG028', 'EEG039', 'EEG050'],
              'resting_stateYF.fif': ['EEG027', 'EEG039', 'EEG062', 'EEG050'],
              'resting_stateYF_2.fif': ['EEG027', 'EEG039', 'EEG050'],
              'resting_stateYO.fif': ['EEG028', 'EEG039', 'EEG050'],
              'resting_stateYO_2.fif': ['EEG027', 'EEG039', 'EEG050'],
              'empty_room.fif': []},
         'W80R0V': 
             {'run01.fif': ['EEG019'], 
              'run02.fif': ['EEG019','EEG048'], 
              'run03.fif': ['EEG019'],
              'run04.fif': ['EEG019'],
              'run05.fif': ['EEG019','EEG039'],
              'run06.fif': ['EEG019','EEG048'],
              'run07.fif': ['EEG019','EEG048'],
              'run08.fif': ['EEG019','EEG048'],
              'run09.fif': ['EEG019','EEG048'],
              'run10.fif': ['EEG019','EEG048'],
              'resting_stateYF.fif': ['EEG019','EEG048'],
              'resting_stateYF_2.fif': ['EEG019','EEG048'],
              'resting_stateYO.fif': ['EEG019','EEG048'],
              'resting_stateYO_2.fif': ['EEG019','EEG048'],
              'empty_room.fif': []},    
         '7TEO5P': 
             {'run01.fif': ['EEG039'], 
              'run02.fif': ['EEG030','EEG039','EEG050'], 
              'run03.fif': ['EEG039','EEG050'],
              'run04.fif': ['EEG039','EEG050'],
              'run05.fif': ['EEG039','EEG050'],
              'run06.fif': ['EEG030','EEG039','EEG050'],
              'run07.fif': ['EEG030','EEG039','EEG050'],
              'run08.fif': ['EEG039','EEG050'],
              'run09.fif': ['EEG039','EEG050'],
              'run10.fif': ['EEG030','EEG039','EEG050'],
              'resting_stateYF.fif': ['EEG039','EEG050'],
              'resting_stateYF_2.fif': ['EEG030','EEG039','EEG050'],
              'resting_stateYO.fif': ['EEG030','EEG039','EEG050'],
              'resting_stateYO_2.fif': ['EEG030','EEG039','EEG050'],
              'empty_room.fif': []},  
        'FADM9A': 
             {'run01.fif': ['EEG029','EEG039'], 
              'run02.fif': ['EEG030','EEG039','EEG050'], 
              'run03.fif': ['EEG039','EEG050'],
              'run04.fif': ['EEG039','EEG041','EEG050'],
              'run05.fif': ['EEG039','EEG041','EEG050'],
              'run06.fif': ['EEG030','EEG039','EEG041','EEG050'],
              'run07.fif': ['EEG030','EEG039','EEG041','EEG050'],
              'run08.fif': ['EEG018','EEG039','EEG041','EEG050','EEG061'],
              'run09.fif': ['EEG039','EEG041','EEG050','EEG061'],
              'run10.fif': ['EEG002','EEG030','EEG039','EEG041','EEG050'],
              'resting_stateYF.fif': ['EEG039','EEG050','EEG058','EEG061'],
              'resting_stateYF_2.fif': ['EEG030','EEG039','EEG041','EEG050'],
              'resting_stateYO.fif': ['EEG019','EEG030','EEG039','EEG041','EEG050','EEG068'],
              'resting_stateYO_2.fif': ['EEG030','EEG039','EEG041','EEG050'],
              'empty_room.fif': []},
        'RGTMSZ':
             {'run01.fif': ['EEG029','EEG039', 'EEG074'], 
              'run02.fif': ['EEG029','EEG039', 'EEG074'], 
              'run03.fif': ['EEG029','EEG039', 'EEG074'],
              'run04.fif': ['EEG029','EEG039', 'EEG074'],
              'run05.fif': ['EEG029','EEG039', 'EEG074'],
              'run06.fif': ['EEG029','EEG039', 'EEG074'],
              'run07.fif': ['EEG029','EEG039', 'EEG074', 'EEG001'],
              'run08.fif': ['EEG029','EEG039', 'EEG074', 'EEG001'],
              'run09.fif': ['EEG029','EEG039', 'EEG074'],
              'run10.fif': ['EEG029','EEG039', 'EEG074', 'EEG001'],
              'resting_stateYF.fif': ['EEG029','EEG039', 'EEG019','EEG074'],
              'resting_stateYF_2.fif': ['EEG029','EEG039','EEG074'],
              'resting_stateYO.fif': ['EEG029','EEG039','EEG074', 'EEG030', 'EEG019'],
              'resting_stateYO_2.fif': ['EEG029','EEG039','EEG074'],
              'empty_room.fif': []},      
        '5P57E6':
             {'run01.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061'], 
              'run02.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061'], 
              'run03.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061', 'EEG050'],
              'run04.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061', 'EEG050'],
              'run05.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061', 'EEG050'],
              'run06.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061', 'EEG050'],
              'run07.fif': ['EEG029','EEG039', 'EEG030', 'EEG040', 'EEG061', 'EEG050'],
              'run08.fif': ['EEG029','EEG039', 'EEG066', 'EEG040', 'EEG061', 'EEG050'],
              'run09.fif': ['EEG018','EEG029','EEG030','EEG039', 'EEG068', 'EEG040', 'EEG061', 'EEG050'],
              'run10.fif': ['EEG018','EEG029','EEG030','EEG039', 'EEG068', 'EEG040', 'EEG061', 'EEG050'],
              'resting_stateYF.fif': ['EEG029','EEG039', 'EEG068', 'EEG040', 'EEG061', 'EEG050'],
              'resting_stateYF_2.fif': ['EEG018','EEG029','EEG030','EEG039', 'EEG068', 'EEG040', 'EEG061', 'EEG050'],
              'resting_stateYO.fif': ['EEG029','EEG039', 'EEG068', 'EEG040', 'EEG061', 'EEG050'],
              'resting_stateYO_2.fif': ['EEG018','EEG029','EEG030','EEG039','EEG040','EEG050','EEG061','EEG068',],
              'empty_room.fif': []},    
        'D1K7TN': 
             {'run01.fif': ['EEG028', 'EEG050', 'EEG061'], 
              'run02.fif': ['EEG050','EEG061'], 
              'run03.fif': ['EEG050','EEG061'],
              'run04.fif': ['EEG050', 'EEG061'],
              'run05.fif': ['EEG050', 'EEG061'],
              'run06.fif': ['EEG050', 'EEG061'],
              'run07.fif': ['EEG050', 'EEG061'],
              'run08.fif': ['EEG050', 'EEG061'],
              'run09.fif': ['EEG050', 'EEG061'],
              'run10.fif': ['EEG050', 'EEG061'],
              'resting_stateYF.fif': ['EEG019', 'EEG050', 'EEG061'],
              'resting_stateYF_2.fif': ['EEG050', 'EEG061'],
              'resting_stateYO.fif': ['EEG050', 'EEG061'],
              'resting_stateYO_2.fif': ['EEG050', 'EEG061'],
              'empty_room.fif': []},   
        'UNI5M7': 
             {'run01.fif': ['EEG039', 'EEG072'], 
              'run02.fif': ['EEG039'], 
              'run03.fif': ['EEG039'],
              'run04.fif': ['EEG039'],
              'run05.fif': ['EEG039'],
              'run06.fif': ['EEG039'],
              'run07.fif': ['EEG039'],
              'run08.fif': ['EEG039'],
              'run09.fif': ['EEG039'],
              'run10.fif': ['EEG039'],
              'resting_stateYF.fif': ['EEG039'],
              'resting_stateYF_2.fif': ['EEG039'],
              'resting_stateYO.fif': ['EEG039'],
              'resting_stateYO_2.fif': ['EEG039'],
              'empty_room.fif': []},                
             
            }    
    
badC_EEG_INCC = {'S01': 
         {1: [], 
          2: [], 
          3: []}}   
    
# run chosen for realignement:
#    2XXX72: run05
#    W80R0V: run08   
#    FADM9A: run06
#    RGTMSZ: run04
#    D1K7TN: run04
#    5P57E6: run06
#    UNI5M7: run01


# ICA components indices to remove (seed=42)
ICA_remove_inds = {'P01': 
             {'block01_2runs.fif': {'mag':[], 'grad':[18,28], 'eeg': [2,3,5,7,8,10,13,14,21,28,44]}, 
              'block02.fif': {'mag':[19], 'grad':[], 'eeg': [11,15,16,18,19,26,28,34,37,44,45,46,48,49]}, 
              'block03.fif': {'mag':[], 'grad':[], 'eeg': [1,11,14,24,28,29,31,36,48,49]},
              'block04.fif': {'mag':[41,46], 'grad':[44], 'eeg': [1,9,10,14,27,28,29,35,36]},
              'block05.fif': {'mag':[], 'grad':[], 'eeg': [0,4,9,17,20,22,23,27,30,34,36,38,43]},
              'block06.fif': {'mag':[], 'grad':[], 'eeg': [8,15,17,21,22,24,25,28,31,40]},
              'block07.fif': {'mag':[36], 'grad':[2], 'eeg': [6,7,10,14,15,19,24,25,33,36, 41]},
              'block08.fif': {'mag':[7], 'grad':[0], 'eeg': [0,4,5,8,12,14,15,17, 21,22,24,25]},
              'block09.fif': {'mag':[], 'grad':[], 'eeg': [0,1,2,3,9,10,12,14,15,16,18,19,30,31]},
              'rest_YF.fif': {'mag':[], 'grad':[], 'eeg': [1,12,14,38,42,47,48]},
              'rest_YF_2.fif': {'mag':[], 'grad':[], 'eeg': [0,1,2,3,5,7,10,15,39,46,47,48]},
              'rest_YO.fif': {'mag':[], 'grad':[], 'eeg': [10,11,13,15,19,22,25,32,49]},
              'rest_YO_2.fif': {'mag':[], 'grad':[], 'eeg': [0,26,31,34,41,48]},
              'empty_room.fif': {'mag':[], 'grad':[]}},
            '2XXX72': 
             {'run01.fif': {'mag':[0,3,4,10], 'grad':[0,7,32, 38], 'eeg': [0,5,6,9,19,21,34]}, 
              'run02.fif': {'mag':[0,3,5,10], 'grad':[0,9,28,37], 'eeg': [0,4,10,17,23, 33, 34, 35, 41, 42, 44, 48]}, 
              'run03.fif': {'mag':[0,4,7,9], 'grad':[0,11,25,42], 'eeg': [5,12,14,17,23,25,28,39,40,45,47]}, 
              'run04.fif': {'mag':[0,3,6,16], 'grad':[0,14,25,40], 'eeg': [0,3, 4,7, 23,29,30,33,37]}, 
              'run05.fif': {'mag':[0,2,8,13], 'grad':[0,6,39,41], 'eeg': [0,6,7,10,12,18,22,23,37,42, 45,46]},  
              'run06.fif': {'mag':[0,2,6,17], 'grad':[0,13,32,39,41], 'eeg': [0,8,11,14, 24,28, 33,37, 38]}, 
              'run07.fif': {'mag':[0,5,6,13], 'grad':[0,13,33,36], 'eeg': [0,6,10, 19,22,27, 31, 32,42]},  
              'run08.fif': {'mag':[0,5,6,7], 'grad':[0,11,36,37], 'eeg': [0,2,6,9,15,25,27,38]}, 
              'run09.fif': {'mag':[0,2,5,18], 'grad':[0,13,29,40], 'eeg': [0,2,4,7,9,14,19,22,24,27,39,43]},
              'run10.fif': {'mag':[0,3,6,14], 'grad':[0,8,37,42], 'eeg': [0,6,7,17,23,24,30,34,41]}, 
              'resting_stateYF.fif': {'mag':[3,39], 'grad':[25,48], 'eeg': [1,2,15,26,31]},  # to rerun
              'resting_stateYF_2.fif': {'mag':[5,24], 'grad':[24,45], 'eeg': [0,6,9,13,14,28,39]},  
              'resting_stateYO.fif': {'mag':[1,3,22], 'grad':[2,9,49], 'eeg': [0,4,5,19,37,40,41,43]},  
              'resting_stateYO_2.fif': {'mag':[0,4,7,16], 'grad':[0,15,36,46], 'eeg': [0,4,5,6,10,13,15,17,22,26,30,40]},  
              'empty_room.fif': {'mag':[], 'grad':[]}},
            'W80R0V': 
             {'run01.fif': {'mag':[0,1,2,3], 'grad':[0,1,2,13], 'eeg': [0,1,2,3,9,20,21,43, 47]}, 
              'run02.fif': {'mag':[0,1,2], 'grad':[0,1,5], 'eeg': [0,1,3,5,13,26,32, 37]}, 
              'run03.fif': {'mag':[0,1,2, 44], 'grad':[0,1,7,5], 'eeg': [0,1,2,6,9,10,11,13,23,29,34,47]}, 
              'run04.fif': {'mag':[0,1,2], 'grad':[0,1,5,32], 'eeg': [0,1,3,24,4,6,15,16,31,44]}, 
              'run05.fif': {'mag':[0,1,2], 'grad':[0,1,2,3,4], 'eeg': [0,1,6,7,10,14,15,17,18, 20,29,30, 31,33]}, 
              'run06.fif': {'mag':[0,1,2,4], 'grad':[0,1,2], 'eeg': [0,1,3,9,12,13,19,28,30,31,34,38]}, 
              'run07.fif': {'mag':[0,1,2], 'grad':[0,1,3], 'eeg': [0,1,2,3,4,6,8,12,16,17,22,24,28,29,41,42,43,44]}, 
              'run08.fif': {'mag':[0,1,2], 'grad':[0,1,2,36], 'eeg': [0,1,2,8,10,15,17,20,30,42]}, 
              'run09.fif': {'mag':[0,1,2], 'grad':[0,1,2,3,4,5,6, 35], 'eeg': [0,1,2,3,4,5,10,11,12,13,27,30,33,34,43]}, 
              'run10.fif': {'mag':[0,1,2,10], 'grad':[0,1,2], 'eeg': [0,1,2,3,4,6,7,8,10,11,12,13,14,20,41,45]}, 
              'resting_stateYF.fif': {'mag':[0,2,3,19], 'grad':[1,2,4,9], 'eeg': [0,4,9,16,23,24,27,29,31,35,36,40,41]}, 
              'resting_stateYF_2.fif': {'mag':[0,1,2], 'grad':[0,1,3,13,49], 'eeg': [0,1,2,5,9,11,15,22,23,26,29,33,37,40,41,42,43]}, 
              'resting_stateYO.fif': {'mag':[0,1,2,4], 'grad':[0,1,2,34], 'eeg': [0,2,3,4,6,15,18,25,27,28,31,35,36,44,45,48]}, 
              'resting_stateYO_2.fif': {'mag':[0,1,2,47], 'grad':[0,1,2,3,42], 'eeg': [0,1,9,14,16,21,22,28,31,34,35,37,38,48]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}},     
            '7TEO5P': 
             {'run01.fif': {'mag':[2,19,29], 'grad':[7,44,48], 'eeg': [0,8,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39]}, 
              'run02.fif': {'mag':[7,24,31], 'grad':[15,48], 'eeg': [0,13,18,23,26,27,28,29,31,32,33,34,35,37,38,40,41,42,43]}, 
              'run03.fif': {'mag':[6,24,32], 'grad':[15,45,46], 'eeg': [0,8,18,22,23,24,25,26,27,28,29,30,31,34,35,36,37,38,39]}, 
              'run04.fif': {'mag':[5,25,29], 'grad':[11,44,45], 'eeg': [0,8,15,17,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,38,39]}, 
              'run05.fif': {'mag':[6,31,32], 'grad':[17,47], 'eeg': [1,14,17, 19,21,22,23,24,27,29,30,33,34,35,36,37,38,39,40,41]}, 
              'run06.fif': {'mag':[8,26,33], 'grad':[18,43,48], 'eeg': [1,13,20,26,28,29,30,31,33,34,35,36,37,38,40,41,42,43,44,45]}, 
              'run07.fif': {'mag':[9,35], 'grad':[20,35], 'eeg': [0,26,27,28,29,30,33,34,35,36,37,38,39,40,41,42]}, 
              'run08.fif': {'mag':[3,17,28], 'grad':[8,31,46], 'eeg': [0,1,3,5,8,10,11,14,18,19,23,24,25,26,27,29,31,34,36,37,38]}, 
              'run09.fif': {'mag':[7,33,35], 'grad':[19,40], 'eeg': [0,14,18,20,21,22,24,26,27,28,29,30,31,32,33,35,37,38,39]}, 
              'run10.fif': {'mag':[7,29,34], 'grad':[13,37,44], 'eeg': [0,9,18,19,23,25,26,27,29,30,31,32,33,34,35,36,37,39,40]}, 
              'resting_stateYF.fif': {'mag':[33], 'grad':[49], 'eeg': [32,34,35,37,38,41,42,43,44]}, 
              'resting_stateYF_2.fif': {'mag':[31,41], 'grad':[44,46], 'eeg': [7,20,22,25,26,27,29,30,32,33,34,35,36,38,39]}, 
              'resting_stateYO.fif': {'mag':[15,31], 'grad':[28,48], 'eeg': [1,21,24,27,28,31,33,34,36,37,38,39,40,44]}, 
              'resting_stateYO_2.fif': {'mag':[11,39], 'grad':[22,47], 'eeg': [1,17,22,25,26,27,28,29,30,31,32,33,34,37,38,39,40]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}},
            'FADM9A': 
             {'run01.fif': {'mag':[0,5,7], 'grad':[4,24,36,38], 'eeg': [0,1,3,6,8,16,20,23,27,29,31,34,35,36,41,42,43,44,48]}, 
              'run02.fif': {'mag':[2,4,7,9], 'grad':[9,40,25,43], 'eeg': [1,3,10,18, 22,28,34,38,39, 45,47,49]}, 
              'run03.fif': {'mag':[1,3,6,29], 'grad':[3,24,36, 49], 'eeg': [17,20,24, 35,37,40,46]}, 
              'run04.fif': {'mag':[1,2,], 'grad':[5,6], 'eeg': [1,5,9, 19,22,28,29,31,35,43,47,48]}, 
              'run05.fif': {'mag':[1,2,4], 'grad':[3,5,39], 'eeg': [0, 1,6,11,13,15,16, 20,31,32,34,39,40,44,45,46,47]}, 
              'run06.fif': {'mag':[0,2,30], 'grad':[3,12], 'eeg': [0,1,2,3,4,5,6,9,10,11,16,17,15,20,26,31,32,33,34,35,39,44]}, 
              'run07.fif': {'mag':[1,2,26], 'grad':[5,11], 'eeg': [2,3,4,5,6,7,9,10,17, 19,22, 25,32, 35,38,39,42,44,47]}, 
              'run08.fif': {'mag':[0,2,7], 'grad':[2,4,42], 'eeg': [0,1,7,9,21, 24,26,27,28,30,36,37,39,40,41,42,48]}, 
              'run09.fif': {'mag':[0,3,6,19], 'grad':[2,6,46,49], 'eeg': [0,1,4,7, 8,10,14,18,21,25, 28,29,30,33,34, 36,38,39,40,41,45,46,47,48]}, 
              'run10.fif': {'mag':[1,3,19], 'grad':[3,9], 'eeg': [0,2,4, 6,11, 12,13, 14,15,16,22,23,25,29,30,31,32,33,36,37,39,45]}, 
              'resting_stateYF.fif': {'mag':[5], 'grad':[21], 'eeg': [0,1,2,6,9,10,11, 12,13,16,21,26,27,32,34,35,46,]}, 
              'resting_stateYF_2.fif': {'mag':[3,6], 'grad':[6,23], 'eeg': [4,6,9,23,24, 25,28,29, 31,35,36,37,48,49]}, 
              'resting_stateYO.fif': {'mag':[3,18,20], 'grad':[28,36], 'eeg': [0,3,4,11,13,15,16,17,19,26,27,28,29,30,34,40,41,43,]}, 
              'resting_stateYO_2.fif': {'mag':[4,29], 'grad':[5], 'eeg': [4,13, 14, 17,21,24,25,26,27,28, 30,31,32,33, 35,37,43,46]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}},             
            'RGTMSZ': 
             {'run01.fif': {'mag':[0,3,9], 'grad':[0,2,35], 'eeg': [0,2,4,7,8,9,12,14,15,16,17,20,23]}, 
              'run02.fif': {'mag':[0,1,11], 'grad':[0,1,37], 'eeg': [0,5,21,7,12,13,16,18,25,26,28,29,32]}, 
              'run03.fif': {'mag':[0,2,17], 'grad':[1,2], 'eeg': [0,6,26,27, 5,7,9,11,14,15,16,17,18,19,24,25]}, 
              'run04.fif': {'mag':[0,2,19], 'grad':[0,2,26], 'eeg': [0,7,13,27,2,3,11,12,16,18,19,23,24,26,28]}, 
              'run05.fif': {'mag':[0,1,19], 'grad':[0,2,30], 'eeg': [0,6,25, 30,4,5,12,15,24,27,28]}, 
              'run06.fif': {'mag':[0,1], 'grad':[0,1], 'eeg': [0,6,22,3,7,9,12,13,15,17,18,20,27,29]}, 
              'run07.fif': {'mag':[0,2,27], 'grad':[0,29,2], 'eeg': [0,9,13, 16,19,20,23,24,27,30,33,38]}, 
              'run08.fif': {'mag':[0,1,14], 'grad':[0,2], 'eeg': [0,4,21,22,14,17,18,23,24,25,26,33]}, 
              'run09.fif': {'mag':[0,1,14], 'grad':[0,40,1,28], 'eeg': [0,3,7,26,2,9,10,12,13,14,15,18,20,25,27,29,30]}, 
              'run10.fif': {'mag':[0,1,12], 'grad':[0,2,41], 'eeg': [0,6,26,17,19,3,4,7,9,11,15,16,18,20,21,22,23,29,34]}, 
              'resting_stateYF.fif': {'mag':[2], 'grad':[3], 'eeg': [11,15,22,23,24,29]}, 
              'resting_stateYF_2.fif': {'mag':[0], 'grad':[0], 'eeg': [9,17,26, 2,19,20,21,23,25,28,36,38]}, 
              'resting_stateYO.fif': {'mag':[1,6], 'grad':[0,9], 'eeg': [1,6,9,7,10,11,13,14,16,17,18,20,22,23,24,28]}, 
              'resting_stateYO_2.fif': {'mag':[0,1,12], 'grad':[0,1,37], 'eeg': [0,6,16,31,1,7,8,10,11,12,19,20,27,30]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}}  ,  
            '5P57E6': 
             {'run01.fif': {'mag':[0,1,2], 'grad':[0,1], 'eeg': [0,2,10,29, 1,3,15,18,26,27,28,30,33,38]}, 
              'run02.fif': {'mag':[0,1,12], 'grad':[0,1], 'eeg': [0,1,7,12, 8,17,25,27,28,32]}, 
              'run03.fif': {'mag':[0,1,6], 'grad':[0,1], 'eeg': [0,1,6, 2,14, 24,28,30,31,33,34,35]}, 
              'run04.fif': {'mag':[0,4, 1], 'grad':[0, 1], 'eeg': [0,1,9,15,23,28,39]}, 
              'run05.fif': {'mag':[0,17,1], 'grad':[0,1], 'eeg': [0,2,13,22,4,8,12,14,15,16,17,24,27,29,39]}, 
              'run06.fif': {'mag':[0,27,1], 'grad':[0,1], 'eeg': [0,1,35,40,3,6,7,8,9,10,14,18,19,20,23,24,32]}, 
              'run07.fif': {'mag':[0,10,1], 'grad':[0,1], 'eeg': [0,2,38, 5,6,7,9,10,11,12,14,15,16,18,19,21,23,27]}, 
              'run08.fif': {'mag':[0,3,1], 'grad':[0,1], 'eeg': [0,1,17,23,26,34,36]}, 
              'run09.fif': {'mag':[0,9,1], 'grad':[0,1], 'eeg': [0,1,13,5,29]}, 
              'run10.fif': {'mag':[0,14,1], 'grad':[0,1], 'eeg': [0,2,3,5,9,10,18,19,20,21,29,34]}, 
              'resting_stateYF.fif': {'mag':[0], 'grad':[0], 'eeg': [11,8,0,5,9,10,15,21,30,32,33]}, 
              'resting_stateYF_2.fif': {'mag':[0], 'grad':[0], 'eeg': [2,8,7,9,10,11,19,24,27,30,31,33,36]}, 
              'resting_stateYO.fif': {'mag':[0,1], 'grad':[0,2], 'eeg': [0,23,1,4,7,9,10,11,15,16,17,18,19,21,22,24,31]}, 
              'resting_stateYO_2.fif': {'mag':[0,1], 'grad':[0,1], 'eeg': [0,17, 2,6,9,11,12,22,23,29,30,31]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}},   
             'D1K7TN': 
              {'run01.fif': {'mag':[0,2,7], 'grad':[0,25,43], 
                             'eeg': [0,4,10,12,14,24,25,28,]}, 
               'run02.fif': {'mag':[0,3,20], 'grad':[0,20], 
                             'eeg': [0,1,6,14,17,21,25,34]}, 
               'run03.fif': {'mag':[0,2,17], 'grad':[0,15],
                             'eeg': [0,1,5,14,17,19,22,27,35]},
               'run04.fif': {'mag':[0,4,5], 'grad':[0,18,44],
                             'eeg': [0,1,5,13,15,16,18,20,22,26,27,28]},
               'run05.fif': {'mag':[0,1,2], 'grad':[0,21,38],
                             'eeg': [0,1,3,6,7,8,9,10,12,13,17,18,19,21,29,31]},
               'run06.fif': {'mag':[0,1,4], 'grad':[0,20,40],
                             'eeg': [0,3,5,9,10,11,12,14,16,18,21,22,24,25,37]},
               'run07.fif': {'mag':[0,2,3,10], 'grad':[0,15,35,36],
                             'eeg': [0,2,7,9,11,12,13,14,15,16,18,19,20,21,23, 25,27,32]},
               'run08.fif': {'mag':[0,1,2], 'grad':[0,15,41],
                             'eeg': [0,2,8,9,10,11,12,14,17,18,21,22,23,24,25, 28]},
               'run09.fif': {'mag':[0,2,5], 'grad':[0,19,35,40],
                             'eeg': [0,4,5,6,10,11,12,15,20,22,23,26,33,38]},
               'run10.fif': {'mag':[0,2,3,38], 'grad':[0,19,41],
                             'eeg': [0,3,5,10,14,17,19,20,26,28,33,35,40]},
               'resting_stateYF.fif': {'mag':[4], 'grad':[16,35],
                                       'eeg': [2,10,12,13,16,20,24,25]},
               'resting_stateYF_2.fif': {'mag':[2,7], 'grad':[12,21],
                                         'eeg': [1,6,7,9,10,11,12,13,17,20,45]},
               'resting_stateYO.fif': {'mag':[0,1], 'grad':[0,14],
                                       'eeg': [0,6,10,15,16,20,22,28,44]},
               'resting_stateYO_2.fif': {'mag':[0,1], 'grad':[1,17],
                                         'eeg': [0,4,6,8,13,19,22,27,39]},
               'empty_room.fif': {'mag':[], 'grad':[]}},              
            'UNI5M7': 
             {'run01.fif': {'mag':[0,7, 14, 12], 'grad':[0,31], 'eeg': [0,9,7,1,6,10,11,13,14,15,16,20,24,29,34,39]}, 
              'run02.fif': {'mag':[0,2,16,26], 'grad':[0,30,45,42], 'eeg': [0,13,4,7,8,9,10,11,12,15,17,19,23,24,26,31]}, 
              'run03.fif': {'mag':[1,0,34], 'grad':[1,0,5], 'eeg': [0,15,3,5,8,9,11,13,14,16,17,18,26,27,28,29,36]}, 
              'run04.fif': {'mag':[0,16], 'grad':[1,25], 'eeg': [0,11,2,6,7,8,9,10,13,14,15,16,17,25,26,27,32,33]}, 
              'run05.fif': {'mag':[0,3], 'grad':[0,21], 'eeg': [1,16,8,0,2,5,6,11,12,15,17,18,21,22,25,26,27,30,31,34]}, 
              'run06.fif': {'mag':[0,4], 'grad':[0,24], 'eeg': [0,12,1,4,6,7,9,10,11,14,15,16,18,21,22,26,29,30,35,36,37]}, 
              'run07.fif': {'mag':[0,15], 'grad':[0,25], 'eeg': [1,11,0,2,4,5,6,8,10,12,13,16,17,18,20,21,24,25,29,33,34,35]}, 
              'run08.fif': {'mag':[0,6], 'grad':[0,38,29], 'eeg': [0,11,3,4,5,8,9,10,12,13,15,16,20,23,24, 27,28,29,31]}, 
              'run09.fif': {'mag':[1,32,0], 'grad':[39,1,34,0], 'eeg': [0,8,5,9,10,11,13,16,17,18,22,24,25,26,27,32]}, 
              'run10.fif': {'mag':[0,8,7], 'grad':[0,47,29], 'eeg': [0,9,2,5,8,10,13,15,16,17,18,23,25,32,33,35]}, 
              'resting_stateYF.fif': {'mag':[13], 'grad':[48], 'eeg': [15, 25, 28, 30, 31, 34]}, 
              'resting_stateYF_2.fif': {'mag':[23], 'grad':[46], 'eeg': [4, 10,12, 16,17,20,23, 25,29,31, 32, 33,34]}, 
              'resting_stateYO.fif': {'mag':[4, 8], 'grad':[10,14], 'eeg': [0,15,41,2,4,6,8,13,14,20,22,28,31,32,33,34]}, 
              'resting_stateYO_2.fif': {'mag':[0,8], 'grad':[0,22], 'eeg': [0,13,3,7,8,9,11,12,17,18,25,28,30,31,32,33,34,47]}, 
              'empty_room.fif': {'mag':[], 'grad':[]}},      
            'XX': 
             {'run01.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run02.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run03.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run04.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run05.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run06.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run07.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run08.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run09.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'run10.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'resting_stateYF.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'resting_stateYF_2.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'resting_stateYO.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'resting_stateYO_2.fif': {'mag':[], 'grad':[], 'eeg': []}, 
              'empty_room.fif': {'mag':[], 'grad':[]}}               
            }  
    
ICA_remove_inds_INCC = {'S01': 
         {1: [0,3], 
          2: [0,2,4], 
          3: [1,6]}}  

# Epoching
l_freq = {'SSVEP' : 1, '5Hzfilt' : 4}    # Hz
h_freq = {'SSVEP' : 45, '5Hzfilt': 6}    # Hz
tmin_ep = -1        # secs
tmax_ep = +2.5      # secs
decim = 5           # 1000Hz to 200Hz
reject = {'ICM' : dict(grad=4000e-13,  # unit: T / m (gradiometers)
              mag=6e-12,      # unit: T (magnetometers)
              eeg=5e-4,      # unit: V (EEG channels)
              ),
          'INCC':dict(eeg=80e-6)}

# Conditions and trigger
triggers_ses1 = {'full': {'trav':11,'stand':12}, 'quad':{'trav':21,'stand':22}} 
triggers_ses2 = {'trav_out':11,'stand':12, 'fov_out':21,'trav_in':22} 
trig_list = {'session1': triggers_ses1, 'session2': triggers_ses2}


