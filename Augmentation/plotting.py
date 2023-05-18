import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn


def plot_samples_and_augmentation(X: np.ndarray, X_augmented: np.ndarray, cmap: str = 'icefire') -> None:
    """
    Plot original and augmented samples.

    Args:
        X: An array of original samples.
        X_augmented: An array of augmented samples.
        cmap: The colormap to use in plotting.

    Returns:
        None
    """    
    for i, (msmt, msmt_augmented) in enumerate(zip(X,X_augmented)):
        print('This is sample index', i)


        clim_bottom=np.min([np.min(msmt[0]),np.min(msmt[1])])
        clim_top=np.max([np.max(msmt[0]),np.max(msmt[1])])
        
        clim_bottom_aug=np.min([np.min(msmt_augmented[0]),np.min(msmt_augmented[1])])
        clim_top_aug=np.max([np.max(msmt_augmented[0]),np.max(msmt_augmented[1])])

        fig,[[ax1,ax3],[ax2,ax4]]=plt.subplots(2,2, figsize=(4,4))
        im = ax1.imshow(msmt[0],cmap=cmap,vmin=clim_bottom,vmax=clim_top, origin='lower')
        
        ax1.set_title('Original\npotentially blocked\n(B=0)')
        ax1.set_aspect('auto')
        ax2.imshow(msmt[1],cmap=cmap,vmin=clim_bottom,vmax=clim_top,
                  origin='lower')
        ax2.set_title('unblocked\n(B=/=0)')
        ax2.set_aspect('auto')
        
        im = ax3.imshow(msmt_augmented[0],cmap=cmap,vmin=clim_bottom,vmax=clim_top, origin='lower')
        #fig.colorbar(im, ax1)
        ax3.set_title('Augmented\npotentially blocked\n(B=0)')
        ax3.set_aspect('auto')
        ax4.imshow(msmt_augmented[1],cmap=cmap,vmin=clim_bottom,vmax=clim_top,
                  origin='lower')
        ax4.set_title('unblocked\n(B=/=0)')
        ax4.set_aspect('auto')
        
        plt.tight_layout()

        plt.show()

        print('------')
        print('======')
        print('------')


def plot_samples(X: np.ndarray, cmap: str = 'icefire') -> None:
    """
    Plot samples.

    Args:
        X: An array of original samples.
        cmap: The colormap to use in plotting.

    Returns:
        None
    """
    for i, msmt in enumerate(X):
        print('This is sample index', i)


        clim_bottom=np.min([np.min(msmt[0]),np.min(msmt[1])])
        clim_top=np.max([np.max(msmt[0]),np.max(msmt[1])])

        fig,(ax1,ax2)=plt.subplots(2,1, figsize=(2,4))
        im = ax1.imshow(msmt[0],cmap=cmap,vmin=clim_bottom,vmax=clim_top, origin='lower')
        #fig.colorbar(im, ax1)
        ax1.set_title('potentially blocked\n(B=0)')
        ax1.set_aspect('auto')
        ax2.imshow(msmt[1],cmap=cmap,vmin=clim_bottom,vmax=clim_top,
                  origin='lower')
        ax2.set_title('unblocked\n(B=/=0)')
        ax2.set_aspect('auto')

        c_bar_x=1.#

        c_bar_length=0.05
        c_bar_height=0.78

        c_bar_y=0.1

        m = cm.ScalarMappable(cmap=cmap)#cm.Spectral)
        m.set_array([clim_bottom,clim_top])

        cbar_ax = fig.add_axes([c_bar_x,c_bar_y, c_bar_length, c_bar_height])
        fig.colorbar(m, cax=cbar_ax,orientation='vertical',label='Current in a.u.')

        plt.tight_layout()

        plt.show()

        print('------')
        print('======')
        print('------')
        
        