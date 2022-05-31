import pandas as pd
from pathlib import Path
from numpy import median, arange, diff, sqrt, array

lco_path = (Path(__file__).parent.parent.parent / 'data/light_curves/lco/').resolve()

m3_files = [#'TIC8348911-01_20210523_LCO-HAL-M3_gp_10px_bjd-flux-err-am-fwhm.dat',
            'TIC8348911-01_20210523_LCO-HAL-M3_rp_15px_bjd-flux-err-am-fwhm.dat',
            'TIC8348911-01_20210523_LCO-HAL-M3_ip_15px_bjd-flux-err-am-fwhm.dat',
            'TIC8348911-01_20210523_LCO-HAL-M3_zs_15px_bjd-flux-err-am-fwhm.dat']
m3_files = [lco_path / fname for fname in m3_files]

lco1m_files = ['TIC8348911-01_20210305_LCO-McD-1m0_ip_10px_bjd-flux-err-detrended.dat',
             'TIC8348911-01_20210326_LCO-CTIO-1m0_ip_10px_bjd-flux-err-detrended.dat',
             'TIC8348911-01_20210416_LCO-CTIO-1m0_ip_10px_bjd-flux-err-detrended.dat']
lco1m_files = [lco_path / fname for fname in lco1m_files]


def read_lco_data():
    times, fluxes, covs = [], [], []

    for f in m3_files:
        df = pd.read_csv(f, delim_whitespace=True)
        times.append(df.iloc[:, 0].values.copy())
        fl = df.iloc[:, 1].values.copy()
        fluxes.append(fl / median(fl))
        cov = df.iloc[:, 3:].values.copy()
        cov = (cov - cov.mean(0)) / cov.std(0)
        covs.append(cov)

    for f in lco1m_files:
        df = pd.read_csv(f, delim_whitespace=True)
        times.append(df.iloc[:, 0].values.copy())
        fl = df.iloc[:, 1].values.copy()
        fluxes.append(fl / median(fl))
        covs.append(array([[]]))

    wns = [diff(f).std() / sqrt(2) for f in fluxes]
    pbs = 'r i z_s i i i'.split()
    ins = 'M3 M3 M3 LCO1m LCO1m LCO1m'.split()
    piis = list(arange(len(times)))

    return times, fluxes, pbs, wns, covs, ins, piis