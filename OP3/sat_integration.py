import os, copy
import glob
import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr
from multiprocessing.dummy import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()

plt.rcParams.update({'font.family': 'serif',
                     'font.size': 16, 'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     })
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

from OP3 import *

opj = os.path.join

rsr_dir = '/DATA/Satellite/rsr'
figdir = '/DATA/projet/garaba/boa_to_toa/Submerged_to_TOA/fig'

# -----------------
# SENTINEL-2
# -----------------
S2_file = opj(rsr_dir, 'Sentinel-2_MSI_Spectral_Responses.xlsx')
S2A = pd.read_excel(S2_file, sheet_name='Spectral Responses (S2A)').dropna(axis=1)
S2B = pd.read_excel(S2_file, sheet_name='Spectral Responses (S2B)').dropna(axis=1)
S2A.columns.values[0] = 'wavelength'
S2B.columns.values[0] = 'wavelength'
S2A.set_index('wavelength', inplace=True)
S2B.set_index('wavelength', inplace=True)

# -----------------
# SENTINEL-3
# -----------------

S3A_file = opj(rsr_dir, 'S3A_OL_SRF_20160713_mean_rsr.nc4')
S3B_file = opj(rsr_dir, 'S3B_OL_SRF_0_20180109_mean_rsr.nc4')
S3A = xr.open_dataset(S3A_file)
S3B = xr.open_dataset(S3B_file)


# -----------------
# WorldView-3
# -----------------
WV3_file = opj(rsr_dir, 'RSR_WV3_VNIR_SWIR.TXT')
header = pd.read_csv(WV3_file, sep='\t', nrows=1, header=None)
header.iloc[:, 0] = "wavelength"
WV3 = pd.read_csv(WV3_file, sep='\t', skiprows=2)
WV3.columns = header.values[0]
WV3 = WV3.set_index('wavelength')  # .to_xarray()

# -----------------
# PACE
# -----------------
PACE_file = opj(rsr_dir, 'pace_oci_RSR.nc')
PACE = xr.open_dataset(PACE_file)
# plt.plot(PACE.wavelength,PACE.RSR.T)

if False:
    # -----------------------
    # Plot RSR of selected satellite missions
    # -----------------------

    colors = ['darkorange', 'goldenrod', 'darkkhaki']
    yname = 0.8
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=0.2, right=0.95, hspace=.15, wspace=0.29)

    i = 0
    axs[i].plot(S3A.mean_spectral_response_function_wavelength.T,
                S3A.mean_spectral_response_function.T, color=colors[i])
    axs[i].plot(S3B.mean_spectral_response_function_wavelength.T,
                S3B.mean_spectral_response_function.T, ':', color=colors[i])
    axs[i].axhline(y=0, lw=2, c='black', clip_on=False)
    axs[i].text(0.0, yname, 'Sentinel 3',
                verticalalignment='bottom', horizontalalignment='right',
                transform=axs[i].transAxes,
                color='black', fontsize=22, zorder=1)
    lines = [plt.Line2D([0], [0], color=colors[i], linewidth=2, linestyle=ls) for ls in ['-', ':']]
    labels = ['S3A', 'S3B']
    axs[i].legend(lines, labels, bbox_to_anchor=(1.02, 0.6), loc='lower right',
                  ncol=1, frameon=False)

    i = 1
    axs[i].plot(S2A.index.values, S2A, color=colors[i], label='S2A')
    axs[i].plot(S2B.index.values, S2B, ':', color=colors[i], label='S2B')
    axs[i].axhline(y=0, lw=2, c='black', clip_on=False)
    axs[i].text(0.0, yname, 'Sentinel 2',
                verticalalignment='bottom', horizontalalignment='right',
                transform=axs[i].transAxes,
                color='black', fontsize=22, zorder=3)
    lines = [plt.Line2D([0], [0], color=colors[i], linewidth=2, linestyle=ls) for ls in ['-', ':']]
    labels = ['S2A', 'S2B']
    axs[i].legend(lines, labels, frameon=False)

    i = 2
    axs[i].plot(WV3.index.values, WV3.drop('PAN', axis=1), color=colors[i])
    axs[i].plot(WV3.index.values, WV3['PAN'], '--', color=colors[i], label='panchromatic')
    axs[i].axhline(y=0, lw=2, c='black', clip_on=False)
    axs[i].legend(frameon=False)
    axs[i].text(0.0, yname, 'WorldView 3',
                verticalalignment='bottom', horizontalalignment='right',
                transform=axs[i].transAxes,
                color='black', fontsize=22, zorder=1)
    for i in range(3):
        axs[i].axis('off')
    axs[i].set_xlim([390, 1100])
    axs[i].axis('on')
    axs[i].set_frame_on(False)
    axs[i].get_xaxis().tick_bottom()
    axs[i].yaxis.set_visible(False)

    axs[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))

    axs[i].set_xlabel('Wavelength (nm)')
    plt.savefig(opj(figdir, 'RSR_vs_wl.png'), dpi=300)

    plt.show()

# ------------------------------
# RSR convolution
# ------------------------------
# !!!!!!!!!!!!!!
# need to run submerged2toa.py to load xtoa
# !!!!!!!!!!!!!!
# xtoa = xtoa.isel(aerosol=0)


def rsr_conv(wl, RSR, Rtoa):
    np.trapz(wl, RSR * Rtoa)


# uncertainty Rtoa
wls = xtoa.wl.values
# ----
# WV3
WV3_ = WV3.to_xarray().sel(wavelength=wls)
S2A_ = S2A.to_xarray().sel(wavelength=wls)

S3df=pd.DataFrame(index=wls)
S3df.index.name='wavelength'
for iband in range(21):
    x_ = S3A.sel(band_number=iband)
    wl = x_.mean_spectral_response_function_wavelength
    r_ = x_.mean_spectral_response_function
    S3df[iband]=np.interp(wls,wl,r_)

S3A_ = S3df.to_xarray()

sza = 30
vza = 5
azi = 75


# for i, (name, x_) in enumerate(xtoa.sel(
#         aot_ref=0.2, depth=0.12, sza=sza, vza=vza, azi=azi).groupby('type')):
#     print(name)
#     for spm in x_.spm:
#         Rtoa = x_.squeeze().sel(spm=0).isel(aot=1).toa_refl
#         RSR.PAN
#         rsr_conv(wls, RSR.PAN, Rtoa)

xtoa_ = xtoa.sel(sza=sza, vza=vza, azi=azi,aerosol='maritime')#.isel(depth=0)
df_ = []
for isat,sat in enumerate(['WV3','S2A','S3A']):
    if sat == 'WV3':
        bands=['PAN', 'COASTAL', 'BLUE', 'GREEN', 'YELLOW', 'RED', 'REDEDGE', 'NIR1', 'NIR2']
        RSR = WV3_

    elif sat=='S2A':
        bands=['S2A_SR_AV_B1', 'S2A_SR_AV_B2', 'S2A_SR_AV_B3', 'S2A_SR_AV_B4',
               'S2A_SR_AV_B5', 'S2A_SR_AV_B6', 'S2A_SR_AV_B7', 'S2A_SR_AV_B8',
               'S2A_SR_AV_B8A']
        RSR =S2A_

    elif sat=='S3A':
        bands=range(20)
        RSR=S3A_



    for band in bands:
        #print(band)
        rsr_ = RSR[band]
        wlmin, wlmax = rsr_[rsr_ > 0.001].wavelength.values[[0, -1]]
        wlc = (wlmax+wlmin)/2
        norm = np.trapz(rsr_, wls)

        for type, xdata in xtoa_.groupby('type'):
            for spm, _xdata in xdata.groupby('spm'):
                for aot_ref, _xdata_ in _xdata.groupby('aot_ref'):
                    for depth,xdata_ in _xdata_.groupby('depth'):
                        #print(aot_ref)
                        xdata_ = xdata_.squeeze()
                        xdata_ = xdata_.dropna('aot',how='all')
                        # if len(xdata_.aot) == 0:
                        #     print(type, spm,depth)
                        #     # plt.close()
                        #     continue
                        xdata_['toa_noadj']=xdata_['toa_refl']-xdata_['bg_refl']
                        Rtoa = xdata_.toa_noadj
                        rband = np.zeros(3)
                        for iaot in range(3):
                            integ = (rsr_ * Rtoa.isel(aot=iaot).values)
                            integ = integ.interpolate_na('wavelength')
                            rband[iaot] = np.trapz(integ, wls) / norm

                        df = pd.Series({'type': type, 'spm': spm, 'depth': depth, 'sat': sat, 'band': band,'aot_ref':aot_ref,
                                        'wlc': wlc, 'wlmin': wlmin, 'wlmax': wlmax,
                                        'Rtoa': rband[1], 'Rtoa_min': rband[0], 'Rtoa_max': rband[2]})
                        df_.append(df)

df = pd.concat(df_, axis=1).T

xsat = df.set_index(['type','spm','depth','aot_ref','sat','band']).to_xarray()


#-------------------------
# plot satellite bands
#-------------------------

colors = ['navy', 'forestgreen', "firebrick"]
sats=['Sentinel-3','Sentinel-2','WorldView-3']
for ispm in range(3):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.125, top=0.96,left=0.05, wspace=0.05, hspace=0.05,right=0.98)
    for isat,sat in enumerate(['S3A','S2A','WV3']):

        xsat_ = xsat.sel(sat=sat).isel(spm=ispm,aot_ref=1)


        for i, (type,xsat__) in enumerate(xsat_.groupby('type')):
            if isat==0:
                axs[isat,i].set_title(type)
            for idepth, (depth, x_) in enumerate(xsat__.groupby('depth')):
                print(type)
                yerr = np.array((x_.Rtoa_max - x_.Rtoa_min).values )
                axs[isat,i].errorbar(np.array(x_.wlc), np.array(x_.Rtoa), yerr=yerr,
                                fmt='.',color=colors[idepth],ecolor=colors[idepth],  capsize=2)

                axs[isat,i].hlines(x_.Rtoa, x_.wlmin, x_.wlmax, lw=2., alpha=0.25,color=colors[idepth],label='z='+str(depth)+' m')
            axs[isat,i].minorticks_on()


    axs[-1,1].legend(loc='upper left', bbox_to_anchor=(0.25, -0.215),
                  fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=17)
    for i in range (4):
        axs[-1,i].set_xlabel('$Wavelength\ (nm)$')
    for isat in range(3):
        axs[isat,0].set_ylabel('$R^{TOA}\ (-)$')
        axs[isat,0].text(0.975, 0.96, sats[isat], size=15,
                                      transform=axs[isat,0].transAxes, ha="right", va="top",
                                      bbox=dict(boxstyle="round",
                                                ec=(0.1, .1, .1),fc=(0.9, .9, .9)
                                                ))

    figname = 'Rtoa_satllite_missions_aot' + str(
                        x_.aot_ref.values) +'_spm'+str(xsat_.spm.values)+ '_sza' + str(
                        sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'
    plt.savefig(opj(figdir, 'toa', figname), dpi=300)

    plt.show()

types=xsat.type.values
for itype in range(len(types)):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 10), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.125, top=0.96,left=0.05, wspace=0.05, hspace=0.05,right=0.98)
    for isat,sat in enumerate(['S3A','S2A','WV3']):
        for i,(spm,xsat_) in enumerate(xsat.sel(sat=sat).isel(aot_ref=1,type=itype).groupby('spm')):

            if isat==0:
                axs[isat,i].set_title('SPM = ' + str(spm) + ' mg/L')
            for idepth, (depth, x_) in enumerate(xsat_.groupby('depth')):

                yerr = np.array((x_.Rtoa_max - x_.Rtoa_min).values)
                axs[isat,i].errorbar(np.array(x_.wlc), np.array(x_.Rtoa), yerr=yerr,
                                fmt='.',color=colors[idepth],ecolor=colors[idepth],  capsize=2)

                axs[isat,i].hlines(x_.Rtoa, x_.wlmin, x_.wlmax, lw=2., alpha=0.25,color=colors[idepth],label='z='+str(depth)+' m')
            axs[isat,i].minorticks_on()


    axs[-1,1].legend(loc='upper left', bbox_to_anchor=(0., -0.215),
                  fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=17)
    for i in range (3):
        axs[-1,i].set_xlabel('$Wavelength\ (nm)$')
    for isat in range(3):
        axs[isat,0].set_ylabel('$R^{TOA}\ (-)$')
        axs[isat,0].text(0.975, 0.96, sats[isat], size=15,
                                      transform=axs[isat,0].transAxes, ha="right", va="top",
                                      bbox=dict(boxstyle="round",
                                                ec=(0.1, .1, .1),fc=(0.9, .9, .9)
                                                ))

    figname = 'Rtoa_'+types[itype]+'turbi_satellite_missions_aot' + str(
                        x_.aot_ref.values) + '_sza' + str(
                        sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'
    plt.savefig(opj(figdir, 'toa', figname), dpi=300)

plt.show()

