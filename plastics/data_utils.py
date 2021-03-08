import os, copy
import glob
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plastics import *

plt.ioff()
plt.rcParams.update({'font.size': 16})


class data:
    def __init__(self, files):
        self.files = files
        pass

    def load(self):
        '''Filename Format
        Polymer Type_Angle_dry or wet_Percentage_Ocean(O) of New(N)_sm_med.dat

        File Headers
        Wavelength | Median Reflectance
        nm               | 100% | 90% |  80% | 70% |  60% | 40% | 20% | 10% | 1% | 0.1% | 0.01%'''
        df_ = []
        for file in self.files:
            params = os.path.basename(file).replace('.dat', '').split('_')
            print(params)
            df = pd.read_csv(file, header=None, sep='\s+')
            df.columns = ['wl', 100, 90, 80, 70, 60, 40, 20, 10, 1, 0.1, 0.01]
            df.dropna(how='all', axis=1)
            df = df.set_index('wl').stack().reset_index()
            df.columns = ['wl', 'concentration', 'hdrf']

            # remove wl when atmo transmittance ~ 0 or low quality
            df = df[df.wl > 450]

            for wl_ in ([1304, 1465], [1763, 1982]):
                df = df[(df.wl < wl_[0]) | (df.wl > wl_[1])]
            df['type'] = params[0]
            vza = float(params[1].replace('deg', ''))
            df['vza'] = vza
            df['condition'] = params[2]
            df['param'] = params[-1]

            df_.append(df)
        df = pd.concat(df_)

        df.set_index(['type', 'condition', 'concentration', 'param', 'vza', 'wl'], inplace=True)
        df = df.unstack('param')
        df.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df.columns]  # df.columns.droplevel(0)
        df['rel_std'] = df.loc[:, 'hdrf_stdev'] / df.loc[:, 'hdrf_avg']
        dff = df  # df[df['rel_std']<1]
        return dff.to_xarray()


class plot:
    def __init__(self, xdata, cond='wet'):
        self.xdata = xdata
        self.cond = cond
        pass

    def spectra(self):
        for conc in self.xdata.concentration.values:
            g = self.xdata.sel(concentration=conc, condition=self.cond).plot.scatter(x='wl', y='hdrf_avg', hue='vza',
                                                                                     col="type", s=2,
                                                                                     col_wrap=3, alpha=0.7,
                                                                                     cmap=plt.cm.Spectral_r,
                                                                                     figsize=(16, 10))
            g.set_xlabels('Wavelength (nm)')
            plt.savefig(
                opj(figdir, 'spectra', 'hdrf_spectra_' + self.cond + '_vs_vza_concentration' + str(conc) + '.png'),
                dpi=300)

            plt.close()

        for vza in self.xdata.vza.values:
            g = self.xdata.sel(vza=vza, condition=self.cond).plot.scatter(x='wl', y='hdrf_avg', hue='concentration',
                                                                          col="type", s=2,
                                                                          col_wrap=3, alpha=0.7, cmap=plt.cm.Spectral_r,
                                                                          figsize=(16, 10))
            g.set_xlabels('Wavelength (nm)')
            plt.savefig(
                opj(figdir, 'spectra', 'hdrf_spectra_' + self.cond + '_vs_concentration_vza' + str(vza) + '.png'),
                dpi=300)

            plt.close()

    def concentration(self, vza, wls=[670, 865, 1020, 1600, 2200]):
        cmap = plt.cm.Spectral_r


        fig, axes = plt.subplots(3, 3, figsize=(20, 12))
        fig.subplots_adjust(top=0.92, left=0.07, right=0.98,
                            hspace=0.3, wspace=0.3)
        axs = axes.ravel()
        i = 0

        for type, group in self.xdata.sel(wl=wls, vza=vza, condition=self.cond).groupby('type'):
            print(i, type)
            g = group.to_dataframe().reset_index()
            c = g.wl

            axs[i].errorbar(g.concentration, g.hdrf_avg, yerr=g.hdrf_stdev, marker='', ls='', ecolor='black',
                            zorder=0)
            im = axs[i].scatter(g.concentration, g.hdrf_avg, c=c, cmap=cmap, alpha=0.95, edgecolors='black')
            #axs[i].scatter(g.concentration, g.hdrf_med, c=c, cmap=cmap, alpha=0.5)

            axs[i].set_title(type)
            i += 1
        axs[0].set_ylabel('HRDF')
        axs[3].set_ylabel('HRDF')
        axs[6].set_ylabel('HRDF')
        axs[6].set_xlabel('Concentration (%)')
        axs[7].set_xlabel('Concentration (%)')
        axs[8].set_xlabel('Concentration (%)')

        cbar = fig.colorbar(im, ax=axes[:, :], shrink=0.9)
        cbar.ax.set_ylabel('Wavelength (nm)')
        plt.savefig(opj(figdir, 'concentration', 'hdrf_vs_concentration_' + self.cond + '_vza' + str(vza) + '.png'),
                    dpi=300)
        plt.close()

    def viewing_angle(self, concentration, wls=[670, 865, 1020, 1600, 2200]):
        cmap = plt.cm.Spectral_r


        fig, axes = plt.subplots(3, 3, figsize=(20, 12))
        fig.subplots_adjust(top=0.92, left=0.07, right=0.98,
                            hspace=0.3, wspace=0.3)
        axs = axes.ravel()
        i = 0

        for type, group in self.xdata.sel(wl=wls, concentration=concentration, condition=self.cond).groupby('type'):
            print(i, type)
            g = group.to_dataframe().reset_index()
            c = g.wl

            axs[i].errorbar(g.vza, g.hdrf_avg, yerr=g.hdrf_stdev, marker='', ls='', ecolor='black',
                            zorder=0)
            im = axs[i].scatter(g.vza, g.hdrf_avg, c=c, cmap=cmap, alpha=0.95, edgecolors='black')
            #axs[i].scatter(g.concentration, g.hdrf_med, c=c, cmap=cmap, alpha=0.5)

            axs[i].set_title(type)
            i += 1
        axs[0].set_ylabel('HRDF')
        axs[3].set_ylabel('HRDF')
        axs[6].set_ylabel('HRDF')
        axs[6].set_xlabel('VZA (deg)')
        axs[7].set_xlabel('VZA (deg)')
        axs[8].set_xlabel('VZA (deg)')

        cbar = fig.colorbar(im, ax=axes[:, :], shrink=0.9)
        cbar.ax.set_ylabel('Wavelength (nm)')
        plt.savefig(opj(figdir, 'vza', 'hdrf_vs_vza_' + self.cond + '_concentration' + str(concentration) + '.png'),
                    dpi=300)
        plt.close()