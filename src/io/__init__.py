import xarray as xa
import pandas as pd

from numpy import array

from .lco import read_lco_data
from .muscat2 import read_m2_data
from .tess import read_tess_data
from .hipercam import read_hipercam_data

def load_mcmc(fname: str):
    ds = xa.load_dataset(fname)
    return pd.DataFrame(array(ds.mcmc_samples).reshape([-1, ds.parameter.size]), columns=ds.parameter)