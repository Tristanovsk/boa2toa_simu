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
                     'font.size': 18, 'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     })
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

from OP3 import *

SixSexe_path = os.path.abspath('6SV1.1/sixsV1.1')
from Py6Sperso import *

from OP3 import data_utils as du

import RTxploitation as RT

RTp = RT.parameterization
ad = RT.auxdata

plot = False

# -----------------------------------
#      Data formatting
# -----------------------------------
idir = '/DATA/projet/garaba/boa_to_toa/Submerged_to_TOA/'
figdir = opj(idir, 'fig')

# -- load blank measurements

res = []
spm = {'LowTurb': 0., 'MedTurb': 75., 'HiTurb': 321.33}
NIR_offset = {'LowTurb': 0.004, 'MedTurb': 0.002, 'HiTurb': 0.0024}

for file in glob.glob(opj(idir, 'data', 'water', 'Water*.dat')):
    name = os.path.basename(file)
    type = name.split('_')[1]
    print(type)
    blank = pd.read_csv(file, sep='\s+')
    blank.columns = ['wl', 'Rmedian', 'Rmean', 'Rstdev']

    blank['Rmedian'] = blank['Rmedian'] - NIR_offset[type]
    blank['Rmean'] = blank['Rmean'] - NIR_offset[type]

    blank['spm'] = spm[type]
    blank['file'] = name
    res.append(blank)

blank = pd.concat(res)
blank_stat = blank.groupby(['spm', 'wl']).mean()
blank_stat['R_sd'] = blank.groupby(['spm', 'wl']).Rmean.std()

blank_xr = blank_stat.to_xarray()

blank_xr = blank_xr.where((blank_xr.wl > 350) & (blank_xr.wl < 1350), drop=True)

colors = ['royalblue', 'grey', 'chocolate']
fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.15, right=0.95,
                    hspace=0.17, wspace=0.17)
axs.minorticks_on()

for i, (spm, x_) in enumerate(blank_xr.groupby('spm')):
    print(spm)
    sd = x_.R_sd
    if spm == 0:
        sd = x_.Rstdev
    axs.plot(x_.wl, x_.Rmean, c=colors[i], label='SPM = ' + str(spm) + ' mg/L')
    axs.fill_between(x_.wl, (x_.Rmean - sd), (x_.Rmean + sd), color=colors[i], alpha=0.3)
axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('$R_{blank}\ (-)$')
plt.legend()
# plt.suptitle('Blanks')

plt.savefig(opj(idir, 'fig', 'above_water', 'blanks.png'), dpi=300)
plt.close()

# -- load full measurements
df = pd.read_excel(opj(idir, 'data', 'Garaba_Dec2019_v2.xlsx'), sheet_name=1,
                   index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8])
df.columns = df.columns.str.replace(' \[nm]', '').astype('float')
df.columns.name = 'wavelength'
dff = (df
       .dropna(how='all')
       .stack()
       .rename('Rrs')
       .reset_index()
       )
dff.columns = dff.columns.str.replace(' ', '_')
dff.columns = dff.columns.str.replace('Total_Suspended_Material_\[mg/L\]', 'SPM')
dff.columns = dff.columns.str.replace('Depth_\[m\]', 'Depth')

# dff=dff.set_index(['Sample_Name', 'Sample_ID', 'Sample_Condition', 'Date/Time',
#        'Latitude', 'Longitude', 'SPM',
#        'Depth', 'wavelength','Product']).unstack()
dff = dff.set_index(['Sample_Name', 'Sample_Condition',
                     'SPM',
                     'Depth', 'wavelength', 'Product']).unstack()
dff.columns = ['_'.join(col) for col in dff.columns]
dff.columns = [col.replace('Standard Deviation', 'SD') for col in dff.columns]

xdata_full = dff.loc[:, ('Rrs_Mean', 'Rrs_Median', 'Rrs_SD')].to_xarray()
xdata = xdata_full.__deepcopy__()

# -----------------------------------
# Plot spectral raw data
# -----------------------------------
colors = ['blue', 'sienna', 'orange', 'grey']
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     "grey", 'forestgreen', 'yellowgreen',
                                                     "khaki", "gold", "darkgoldenrod",
                                                     'orangered', "firebrick", 'purple'])
norm = mpl.colors.Normalize(vmin=min(xdata.Depth), vmax=1)  # max(xdata.Depth))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
labels=['(a)','(b)','(c)','(d)']
param = 'Rrs_Mean'
abs_feature = [931, 1045, 1215, 1417, 1537, 1732, 2046, 2313]
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 10), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.07, right=0.95,
                    hspace=0.17, wspace=0.05)
axs = axs.ravel()
# raw wet plastics
x_ = xdata.sel(Sample_Condition='Wet').sel(Depth=0).dropna('SPM').squeeze()
for i, (name, x__) in enumerate(x_.groupby('Sample_Name')):
    axs[0].plot(x__.wavelength, x__[param], lw=2, color=colors[i], label=name)
    axs[0].fill_between(x__.wavelength, x__[param] - x__['Rrs_SD'], x__[param] + x__['Rrs_SD'], alpha=0.3,
                        color=colors[i])
axs[0].vlines(abs_feature, 0, 0.8, ls=':', color='grey')

x_ = xdata.sel(Sample_Condition='Wet').isel(Depth=1).squeeze()
for ispm, (spm, x__) in enumerate(x_.groupby('SPM')):
    axs[ispm + 1].set_title(labels[ispm+1]+' subsurface - SPM = ' + str(spm) + ' mg/L')
    for i, (name, x___) in enumerate(x__.groupby('Sample_Name')):
        axs[ispm + 1].plot(x___.wavelength, x___[param], lw=2, color=colors[i], label=name)
        axs[ispm + 1].fill_between(x___.wavelength, x___[param] - x___['Rrs_SD'], x___[param] + x___['Rrs_SD'],
                                   alpha=0.3, color=colors[i])
axs[-1].legend(fancybox=True, shadow=True)
axs[0].set_ylabel('$Reflectance\ (-)$')
axs[2].set_ylabel('$Reflectance\ (-)$')
axs[0].set_title('(a) Wet - Above water')
axs[2].set_xlabel('$Wavelength\ (nm)$')
axs[3].set_xlabel('$Wavelength\ (nm)$')

for i in range(4):
    axs[i].minorticks_on()

plt.savefig(opj(idir, 'fig', 'above_water', 'lab_Rrs_full_spectral_range.png'), dpi=300)

# -----------------------------------
# VisNIR spectral range
# -----------------------------------

wls = xdata_full.wavelength
xdata = xdata_full.sel(wavelength=wls[(wls > 350) & (wls < 1300)])

aw, bbw = ad.iopw().get_iopw(wls)


# -----------------------------
# fit with exponential decay with depth
# -----------------------------

# -----------------------------------
# Zmax vs wavelength
def zmax(A, B, R_blank,eps=0.01):
    return 1. / B * np.log(A/eps)


def exp_fit(depth, a, b, c):
    return a * np.exp(-b * depth) + c


param = 'Rrs_Mean'
file_fit = opj(idir, 'data', 'exp_fit_with_depth_v2.csv')
if not os.path.exists(file_fit):
    res = []
    for i, (name, x_) in enumerate(
            xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).groupby('Sample_Name')):
        print(name)
        for wl_ in x_.wavelength:
            blank_ = blank_xr.sel(wl=wl_)
            x__ = x_.sel(wavelength=wl_)
            for ii, (spm, x___) in enumerate(x__.groupby('SPM')):
                x, y = x___.Depth, x___[param]
                if np.isnan(y.values).any():
                    continue
                R_blank = blank_.sel(spm=spm)['Rmean'].values
                p0 = 0.001, 1, R_blank
                try:
                    popt, pcov = so.curve_fit(exp_fit, x, y, p0=p0,
                                              bounds=([-1, 0, R_blank * 0.95], [1, 251., R_blank * 1.05]))
                except:
                    popt = [np.nan] * 3
                z_max = zmax(*popt)
                perr = np.sqrt(np.diag(pcov))
                res.append([name, spm, float(wl_), *popt, *perr,z_max])

    res_df = pd.DataFrame(res, columns=['name', 'spm', 'wl', 'A', 'B', 'C', 'Asd', 'Bsd','Csd','Zmax'])
    res_df.to_csv(file_fit, index=False)
else:
    res_df = pd.read_csv(file_fit)

res_xr = res_df.set_index(['name', 'spm', 'wl']).to_xarray()

# -----------------------------
# fit R_plastic - R_blank with exponential decay with depth
# -----------------------------
#
# param = 'Rrs_Mean'
#
# def exp2_fit(depth, a, b):
#     return a * np.exp(-b * depth)
#
#
# file_fit = opj(idir, 'data', 'exp_fit_with_depth.csv')
# if not os.path.exists(file_fit):
#     res = []
#     for i, (name, x_) in enumerate(
#             xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).groupby('Sample_Name')):
#         print(name)
#         for wl_ in x_.wavelength:
#             blank_ = blank_xr.sel(wl=wl_)
#             x__ = x_.sel(wavelength=wl_)
#             for ii, (spm, x___) in enumerate(x__.groupby('SPM')):
#                 x, y = x___.Depth, x___[param]
#                 if np.isnan(y.values).any():
#                     continue
#                 R_blank = blank_.sel(spm=spm)['Rmean'].values
#                 y = y - R_blank
#                 p0 = 0.001, 1
#                 try:
#                     popt, pcov = so.curve_fit(exp2_fit, x, y, p0=p0,
#                                               bounds=([-0.01, 0], [1, 251.]))
#                 except:
#                     popt = [np.nan] * 3
#
#                 res.append([name, spm, float(wl_), *popt])
#
#     res_df = pd.DataFrame(res, columns=['name', 'spm', 'wl', 'A', 'B'])
#     res_df.to_csv(opj(idir, 'data', 'exp2_fit_with_depth.csv'), index=False)
# else:
#     res_df = pd.read_csv(opj(idir, 'data', 'exp2_fit_with_depth.csv'))
#
# res_xr = res_df.set_index(['name', 'spm', 'wl']).to_xarray()

# -----------------------------------
# PLOTTING
# -----------------------------------

if plot:

    # Fitting parameters
    res_xr.plot.scatter(x='wl', y='A', hue='spm', col='name', col_wrap=2)
    res_xr.plot.scatter(x='wl', y='B', hue='spm', col='name', col_wrap=2)
    res_xr.plot.scatter(x='wl', y='C', hue='spm', col='name', col_wrap=2)

    # -----------------------------------
    # spectral raw data
    norm = mpl.colors.Normalize(vmin=min(xdata.Depth), vmax=1)  # max(xdata.Depth))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(17, 13), sharey=True, sharex=True)
    fig.subplots_adjust(bottom=0.15, top=0.962, left=0.086, right=0.98,
                        hspace=0.05, wspace=0.05)
    # axs = axs.ravel()
    param = 'Rrs_Mean'
    for i, (name, x_) in enumerate(
            xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).groupby('Sample_Name')):
        print(name)
        for ii, (spm, x__) in enumerate(x_.groupby('SPM')):
            xfit = res_xr.sel(name=name, spm=spm)

            if np.isnan(x__[param].values).all():
                axs[i, ii].set_axis_off()
                print(i, ii)
                continue
            # measured depths
            axs[i, ii].minorticks_on()
            for depth, x___ in x__.groupby('Depth'):
                p = axs[i, ii].plot(x___.wavelength, x___[param], color=cmap(norm(depth)),
                                    label=str(depth) + ' m',
                                    zorder=0)
            # extra depth from fitting
            # print('zmax',zmax(xfit.A, xfit.B, xfit.C,))
            for depth in [0.2, 0.33, 0.4, 0.45, 0.5, 0.6, 0.75, 1.5]:
                simu = exp_fit(depth, xfit.A, xfit.B, xfit.C)
                p = axs[i, ii].plot(simu.wl, simu, '--', color=cmap(norm(depth)),
                                    label=str(depth) + ' m',
                                    zorder=0)

    for irow, sample in enumerate(xdata.Sample_Name.values):
        axs[irow][0].text(0.95, 0.95, sample, size=15,
                          transform=axs[irow][0].transAxes, ha="right", va="top",
                          bbox=dict(boxstyle="round",
                                    ec=(0.1, 0.1, 0.1),
                                    fc=plt.matplotlib.colors.to_rgba(colors[irow], 0.2)))
        axs[irow][0].set_ylabel('$R_{plastic}\ (-)$')
    for ii, spm in enumerate(xdata.SPM.values):
        axs[0, ii].set_title('SPM = ' + str(spm) + ' mg/L')
        axs[-1, ii].set_xlabel('$Wavelength\ (nm)$')
    axs[-1, -2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                       fancybox=True, shadow=True, ncol=7, handletextpad=0.1, fontsize=20)
    plt.savefig(opj(idir, 'fig', 'lab_data_plus_extra_depths_v2.png'), dpi=300)

    # -----------------------------------
    # Zmax vs wavelength
    # -----------------------------------

    color_spm = ["blue", "grey", 'orangered']
    ls = ['-', '--']
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 10), sharex=True,sharey=True)
    fig.subplots_adjust(bottom=0.16, left=0.05, wspace=0.05, right=0.98)
    axs = axs.ravel()
    param = 'Zmax'
    for i, (name, x_) in enumerate(
            res_xr.groupby('name')):
        print(name)
        axs[i].minorticks_on()
        for ii, (spm, x__) in enumerate(x_.groupby('spm')):
            for i_eps, eps in enumerate([0.01, 0.05]):
                z_max = -1 / x__.B * np.log(eps / x__.A)
                z_max= 1/ x__.B*np.log( x__.A * x__.B/eps)

               # z_max[x__.Asd > 1.] =  np.nan
                # z_max[z_max > 6] = np.nan
                z_max[z_max < 0] = 0
                p = axs[i].plot(x__.wl, z_max, ls=ls[i_eps], lw=2, color=color_spm[ii],
                                label='$\epsilon=$' + str(eps) + '; SPM=' + str(spm) + ' mg/L',
                                zorder=0)
        axs[i].set_title(labels[i]+' '+name)


    axs[2].legend(loc='upper center', bbox_to_anchor=(1, -0.175),
                  fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=17)
    axs[2].set_xlabel('$Wavelength\ (nm)$')
    axs[3].set_xlabel('$Wavelength\ (nm)$')
    axs[0].set_ylabel('$z_{max}\ (m)$')
    axs[2].set_ylabel('$z_{max}\ (m)$')
    plt.savefig(opj(idir, 'fig', 'Zmax_plastics.png'), dpi=300)

    # -----------------------------------
    # vs Depth
    # -----------------------------------

    linear = False

    depths = np.linspace(0, 0.5, 100)
    cmap = mpl.cm.Spectral_r
    norm = mpl.colors.Normalize(vmin=400, vmax=1000)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.9,
                        hspace=0.3, wspace=0.2)
    # axs = axs.ravel()
    param = 'Rrs_Mean'
    for i, (name, x_) in enumerate(
            xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).groupby('Sample_Name')):
        for wl_ in [400, 450, 500, 550, 600, 650, 700, 800, 850, 1020]:
            print(name)
            x__ = x_.sel(wavelength=wl_)

            for ii, (spm, x___) in enumerate(x__.groupby('SPM')):
                xfit = res_xr.sel(name=name, spm=spm, wl=wl_)
                x, y = x___.Depth, x___[param]
                if np.isnan(y.values).any():
                    axs[i, ii].set_axis_off()
                    continue

                axs[i, ii].plot(depths, exp_fit(depths, xfit.A.data, xfit.B.data, xfit.C.data), '--',
                                color=cmap(norm(wl_)), zorder=0)

                p = axs[i, ii].plot(x, y, 'o', color=cmap(norm(wl_)), label=str(wl_) + ' nm', zorder=1)

        axs[i][0].set_ylabel('$R_{plastic}\ (-)$')
    for ii, spm in enumerate(xdata.SPM.values):
        axs[0, ii].set_title('SPM = ' + str(spm) + ' mg/L')
        axs[-1, ii].set_xlabel('Depth (m)')
    axs[-1, -2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                       fancybox=True, shadow=True, ncol=5, handletextpad=0.1, fontsize=20)
    if linear:

        for irow, sample in enumerate(xdata.Sample_Name.values):
            axs[irow][0].text(0.95, 0.95, sample, size=15,
                              transform=axs[irow][0].transAxes, ha="right", va="top",
                              bbox=dict(boxstyle="round",
                                        ec=(0.1, 0.1, 0.1),
                                        fc=plt.matplotlib.colors.to_rgba(colors[irow], 0.15)))
        plt.savefig(opj(idir, 'fig', 'lab_data_vs_depth_v2.png'), dpi=300)
    else:
        for i in range(4):
            for j in range(3):
                axs[i, j].semilogx()
                # axs[i, j].semilogy()
        for irow, sample in enumerate(xdata.Sample_Name.values):
            axs[irow][0].text(0.98, 0.95, sample, size=15,
                              transform=axs[irow][0].transAxes, ha="right", va="top",
                              bbox=dict(boxstyle="round",
                                        ec=(0.1, 0.1, 0.1),
                                        fc=plt.matplotlib.colors.to_rgba(colors[irow], 0.15)))
        plt.savefig(opj(idir, 'fig', 'lab_data_vs_depth_log_v2.png'), dpi=300)

    plt.show()

# -----------------------------
# get spectra of reference for VRT calculations
# -----------------------------

rho_w = xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).sel(
    wavelength=wls[(wls >= 350) & (wls <= 1200)]).Rrs_Mean
# reproject on new wavelength wls
wls = np.arange(350, 1200, 5)
rho_w = rho_w.interp(wavelength=wls)

#######
##
#
# END
#
##
#######
