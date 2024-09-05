import numpy as np
import scipy
from toolbox.comparison import complex_corr


def fisherZ(R2):
    '''
    Return the Fisher zscore of a correlation coefficient. Allows to get a
    uniform distribution

    Parameters
    ----------
    R2 : FLOAT or ARRAY
        Coefficient correlation

    Returns
    -------
    z : FLOAT or ARRAY
        Fisher Z-score

    '''
    z = 0.5*np.log((1+R2)/(1-R2)) 
    return z

def fisherZ_toR2(Z):
    '''
    Return the correlation coefficient of a Fisher zscore. Allows to get a
    uniform distribution

    Parameters
    ----------
    Z : FLOAT or ARRAY
        Coefficient correlation

    Returns
    -------
    R2 : FLOAT or ARRAY
        Fisher Z-score

    '''
    R2 = (np.exp(2*Z)-1) /  (np.exp(2*Z)+1)
    return R2
    
    
def create_R2_shuffledDistrib(mat_to_shuffle, mat_meas, N_permut=5000):
    '''
    Create a null distribution of R2 by comparing a vector and shuffled instances
    of a second vector. The permutation is done by shuffling sensors within each 
    condition

    Parameters
    ----------
    mat_to_shuffle : ARRAY of size conditions*sensors
        Vector to shuffle.
    mat_meas : ARRAY of size conditions*sensors
        Vector to compare to each shuffled instance of mat_to_shuffle.      
    N_permut : INT
        Number of permutations. 5000 by default.

    Returns
    -------
    R2_sh : ARRAY
        Permutation distribution of R. 

    '''
    # Upper values of the matrix to compare
    R2_sh = np.zeros((N_permut))
    for n in range(N_permut):
        # Shuffle predicted data within each condition
        mat_shuffled = np.zeros(np.shape(mat_to_shuffle), dtype='complex_')
        for c in range(len(mat_to_shuffle)):
            mat_shuffled[c] = np.random.permutation(mat_to_shuffle[c])

        # Correlate across sensors
        x = mat_shuffled.flatten()
        y = mat_meas.flatten()
        R2_sh[n], z, p = complex_corr(y, x)
    
    return R2_sh


def permutation_test_singleSubj(mat_meas, mat_simu, N_permut):

    # Create shuffled distribution of R2
    R2_sh = create_R2_shuffledDistrib(mat_simu, mat_meas, N_permut)

    # fisher Z to get normal distribution
    zscores_sh = fisherZ(R2_sh)

    # Calculate real R
    R_real, z, p = complex_corr(mat_meas.flatten(), mat_simu.flatten())    

    # Compute associated p-values
    p = (100- scipy.stats.percentileofscore(zscores_sh, z))/100
    
    return R_real, p, R2_sh

def permutation_test_group(mat_meas, mat_simu, N_permut_indiv, N_permut_grp, alpha):
    ''' 
    Permutation-based statistics (permutation across sensors) on the correlation coefficient 
    between measured and predicted data.

    Parameters
    ----------
    mat_meas: ARRAY subjects*conditions*sensors
        Measured data
    mat_simu: ARRAY subjects*conditions*sensors
        Predicted data
    N_permut_indiv: INT
        Number of permutations to create the shuffled distribution for each individual
    N_permut_grp: INT
        Number of permutations to create the shuffled distribution at the group level 
    
    Returns
    ----------

    '''
    N = len(mat_meas)
    # Create each individual null distribution
    print('Individual permutation distributions in progress...')
    R2_sh_indiv = np.zeros((N, N_permut_indiv))
    R2_indiv = np.zeros((N,)); p_indiv = np.zeros((N,))
    for n in range(N):
        R2_indiv[n], p_indiv[n], R2_sh_indiv[n] = permutation_test_singleSubj(mat_meas[n], mat_simu[n], N_permut_indiv)
    
    # Create the group level null distribution
    print('Group permutation distribution in progress...')
    zscores_sh_grp = np.zeros((N_permut_grp))
    for i in range(N_permut_grp):

        new_sample = [np.random.choice(R2_sh_indiv[i]) for i in range(N)]        
        R2_grp = np.mean(new_sample)
        zscores_sh_grp[i] = fisherZ(R2_grp) 
        
    # Calculate the true group level R2 and the associated pvalue
    R2_true = np.mean(R2_indiv)
    zscore_true = fisherZ(R2_true)
    
    p = (100- scipy.stats.percentileofscore(zscores_sh_grp, zscore_true))/100
    
    # Calculate the R2 corresponding to the alpha level
    zscore_alpha = np.percentile(zscores_sh_grp, alpha)
    R2_alpha = fisherZ_toR2(zscore_alpha)
    
    return R2_indiv, p_indiv, p, R2_alpha
        
    