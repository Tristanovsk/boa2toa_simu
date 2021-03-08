import os, glob
opj = os.path.join
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()
plt.rcParams.update({'font.family': 'Times New Roman',
                     'font.size': 16, 'axes.labelsize': 20,
                     'xtick.minor.visible': True,
                     'xtick.major.size': 5,
                     'ytick.minor.visible': True,
                     'ytick.major.size': 5})

from RTxploitation import lutplot
from Py6Sperso import *
from plastics import *

lp = lutplot.plot()
aot = 0.2
aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert]]
aerosol = aerosols[1]
amodel = aerosol[0]
dff=[]
for sza in (5,30,60):
    for wl in (865,1600):
        resfile = 'plastic_dir_impact_boa_toa_amodel_' + amodel + '_aot' + str(aot) + \
          '_sza' + str(sza) + '_wl' + str(wl)
        datafile = opj(idir, 'data', resfile + '.csv')
        dff.append(pd.read_csv(datafile))
dff = pd.concat(dff)
xtoa = dff.set_index(['type', 'xfoam', 'wl','sza','vza','azi']).to_xarray()


# ------------------------------
# Plotting section
# ------------------------------
# ----------------------------
# plot rho factor for given SZA
# ----------------------------
# construct raster dimensions
type='B'

type_label = {'B': 'Sheets', 'Foam': 'Foam', 'HDPE': 'HDPE',
              'Hard': 'Hard Fragments', 'LDPE': 'LDPE', 'P': 'Pellets',
              'PP': 'PP', 'PS': 'PS', 'Rope': 'Rope'}
vmax={'B': 0.2, 'Foam': 0.2, 'HDPE': 0.2,
              'Hard': 0.1, 'LDPE': 0.1, 'P': 0.1,
              'PP': 0.2, 'PS': 0.2, 'Rope': 0.1}
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["darkblue","cornflowerblue","white","gold","red",'firebrick'])#"firebrick","darkgoldenrod",,'purple'
vzamax=80
wl = 865
for sza in (5,30,60):
    xtoa_ = xtoa.sel(sza=sza,wl=wl).interp(azi=np.linspace(0,360,100),method='cubic').interp(vza=np.arange(0,80,1), method='cubic')
    r, theta = np.meshgrid(xtoa_.vza, np.radians(xtoa_.azi))

    for type, group in xtoa_.toa_refl.groupby('type'):
        resfile = 'plastic_dir_impact_boa_toa_amodel_' + amodel + '_aot' + str(aot) + \
          '_sza' + str(sza) + '_wl' + str(wl)
        fig, axs = plt.subplots(1,6, figsize=(26, 8), subplot_kw=dict(projection='polar'))

        for i, xfoam in  enumerate([0,0.2,0.4,0.6,0.8,1.]):
            arr = group.sel(xfoam=xfoam)
            cax = lp.add_polplot(axs[i], r, theta, arr.T,
                                 title=str(xfoam*100)+' %',cmap=cmap,scale=False,
                                 vmin=0.0,vmax=0.2) #vmax[type])
        plt.tight_layout()
        fig.subplots_adjust(top=.85)
        cb = fig.colorbar(cax, ax=axs.ravel().tolist(),
                          shrink=0.6, location='top',pad=0.25,label=type_label[type])
        cb.ax.tick_params(labelsize=19)
        #plt.suptitle(type,fontsize=24)
        plt.savefig(opj(idir, 'fig', resfile + '_'+type+'_polar_plot.png'), dpi=300)
