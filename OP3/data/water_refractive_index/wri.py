''' Executable to process L1C images from Sentinel-2 and Landsat mission series

Usage:
wri <wavelength> <temperature> <salinity>
wri -h | --help
wri -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <wavelength>  wavelength in microns [default: 0.440]
  <temperature>  temperature in degC [default: 20.]
  <salinity>   salinity in PSU [default: 33.]
'''

import os, sys
import numpy as np
from docopt import docopt
sys.path.extend([os.path.join(os.path.dirname(__file__))])
import auxdata as ad

if __name__ == "__main__":

    args = docopt(__doc__)
    #print(args)
    wl = np.array([float(args['<wavelength>'])*1000])

    temp= float(args['<temperature>'])
    sal = float(args['<salinity>'])
    wri = ad.water_refractive_index()

    print( "{:0.5f}".format(wri.n_harmel(wl,temp,sal)[0]))

