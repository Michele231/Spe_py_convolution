
# Author:       Michele Martinazzo
# Institution:  University of Bologna


# Description:  This is a simple function created to reduce the resolution of
#               a spectrum.
#

import numpy as np

def conv_spect(x,y,resolution = 0.3,start_v=None,final_v=None,
                    ftype = 'sinc'):
    ''' 
    '''

    x = np.asarray(x)
    y = np.asarray(y)

    min_diff_x = np.amin(np.diff(x)) # min step of x
    avr_diff_x = np.sum(np.diff(x))/len(np.diff(x))
    
    if start_v==None:
        start_v = np.amin(x)
    if final_v==None:
        final_v = np.amax(x)
        
    # force the start_v, final_v and resolution to be multiple of avr_diff_x
    resolution = np.round(resolution/avr_diff_x)*avr_diff_x
    start_v = np.round(start_v/resolution)*resolution
    final_v = np.round(final_v/resolution)*resolution
    
    # Check values
    if ((avr_diff_x-min_diff_x)>=min_diff_x*0.005):
        raise ValueError(f'Irregular x grid: Dx_min = {min_diff_x}, Dx_ave = {avr_diff_x}')

    if (resolution < min_diff_x):
        raise ValueError('The required spectral sampling interval too low')
    elif ((resolution - min_diff_x)<=min_diff_x*0.01): # within 1% of rel diff
        print('Warning: The input is already at this resolution!')
        print('Returning the original data...')
        selection = np.where((x>=start_v) & (x<=final_v))
        return x[selection], y[selection]
    
    # x low resulution (xl)
    xl = np.linspace(start_v,final_v,
                    int(np.round((final_v-start_v)/resolution))+1)

    # Create the sinc function
    sinc_len = 100*resolution               # +-    

    if sinc_len>=(np.amax(x)-np.amin(x)):
        print('WARNING The x invervall is too short:')
        print('Reducing the dimension of the convolution function...')
        sinc_len = 30*resolution
        if sinc_len>=(np.amax(x)-np.amin(x)):
            raise ValueError(f'x intervall too short for the selected resolution!')

    if ftype == 'sinc':
        q0 = np.linspace(-sinc_len/2,sinc_len/2,int(sinc_len/avr_diff_x)+1)
        sx = np.sinc(q0/resolution)
        # Normalization
        sx = sx/np.sum(sx) 
                             
    elif ftype == 'gauss':
        q0 = np.linspace(-sinc_len/2,sinc_len/2,int(sinc_len/avr_diff_x)+1)
        sx = np.exp(-np.power(q0, 2.) / (2 * np.power(resolution, 2.)))
        # Normalization
        sx = sx/np.sum(sx)
        
    elif ftype == 'square':
        sinc_len = resolution
        q0 = np.linspace(-sinc_len/2,sinc_len/2,int(sinc_len/avr_diff_x)+1)
        sx = np.ones(len(q0))
        # Normalization
        sx = sx/len(sx)

    else:
        raise ValueError(f'The ftype = {ftype} does not exist: sinc, gauss or square!')

    # Convolution
    yl = np.convolve(sx,y,'same') # High res convolved rad
                                  # Mode ‘same’ returns output of length 
                                  # max(M, N). Boundary effects are still visible

    yl = np.interp(xl, x, yl)     # Low res convolved rad !!! the final resolution
                                  # has to be a multiple of the initial resolution

    return xl, yl

