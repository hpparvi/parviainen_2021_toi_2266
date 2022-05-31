import pandas as pd
import xarray as xa
from astropy.units import Rjup, Rsun, AU
from numpy import sqrt, degrees, radians, array, where
from numpy.random.mtrand import normal, uniform
from pytransit.orbits import as_from_rhop, i_from_ba, d_from_pkaiews
from pytransit.utils.eclipses import Teq

# Plotting
# --------
AACW = 3.46  # A&A column width [inch]
AAPW = 7.1   # A&A page width [inch]


N = lambda a: a/a.median()

def normalize(a):
    if isinstance(a, pd.DataFrame):
        return (a - a.mean()) / a.std()

def split_normal(mu, lower_sigma, upper_sigma, size=1):
    z = normal(0, 1, size=size)
    return where(z<0, mu+z*lower_sigma, mu+z*upper_sigma)
    

# Result dataframe routines
# -------------------------

def read_mcmc(fname, flatten=True):
    with xa.open_dataset(fname) as ds:
        if flatten:
            try:
                npt = ds.lm_mcmc.shape[-1]
                df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['name'].values)
            except AttributeError:
                npt = ds.mcmc_samples.shape[-1]
                df = pd.DataFrame(array(ds.mcmc_samples).reshape([-1, npt]), columns=ds.parameter)
            return df
        else:
            try:
                return array(ds.lm_mcmc)
            except AttributeError:
                return array(ds.mcmc_samples)


def read_tess_mcmc(fname):
    with xa.open_dataset(fname) as ds:
        npt = ds.lm_mcmc.shape[-1]
        df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['lm_parameter'].values)
    return df


def derive_qois(df_original, rstar, star_teff):
    df = df_original.copy()
    ns = df.shape[0]

    rstar_d = normal(rstar.n, rstar.s, size=ns) * Rsun
    period = df.p.values

    df['a_st'] = as_from_rhop(df.rho.values, period)
    df['a_au'] = (df.a_st.values * rstar_d.to(AU)).value
    df['inc'] = degrees(i_from_ba(df.b.values, df.a_st.values))
    df['teff_p'] = Teq(normal(star_teff.n, star_teff.s, size=ns), df.a_st, uniform(0.25, 0.50, ns),
                       uniform(0, 0.4, ns))

    if 'k2_app' in df:
        df['k_true'] = sqrt(df.k2_true)
        df['k_app'] = sqrt(df.k2_app)
        df['cnt'] = 1. - df.k2_app / df.k2_true
        df['cnt_tess'] = 1. - df.k2_app_tess / df.k2_true
        df['t14'] = d_from_pkaiews(period, df.k_true.values, df.a_st.values, radians(df.inc.values), 0.0, 0.0, 1)
        df['t14_h'] = 24 * df.t14
        df['r_app'] = df.k_app.values * rstar_d.to(Rearth)
        df['r_true'] = (df.k_true.values * rstar_d.to(Rearth)).value
        df['r_app_rsun'] = df.k_app.values * rstar_d.to(Rsun)
        df['r_true_rsun'] = df.k_true.values * rstar_d.to(Rsun)
    else:
        df['k'] = sqrt(df.k2)
        df['r'] = (df.k.values * rstar_d.to(Rearth)).value
        df['t14'] = d_from_pkaiews(period, df.k.values, df.a_st.values, radians(df.inc.values), 0.0, 0.0, 1)
        df['t14_h'] = 24 * df.t14

    return df

    #df['r_app'] = df.k_app.values * rstar_d.to(Rjup)
    #df['r_true'] = df.k_true.values * rstar_d.to(Rjup)

    #df['r_app_rsun'] = df.k_app.values * rstar_d.to(Rsun)
    #df['r_true_rsun'] = df.k_true.values * rstar_d.to(Rsun)
    #df['teff_p'] = Teq(normal(star_teff.n, star_teff.s, size=ns), df.a_st, uniform(0.25, 0.50, ns), uniform(0, 0.4, ns))
    #return df
