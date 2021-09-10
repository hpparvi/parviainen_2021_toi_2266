import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from numpy import diff, sqrt, arange, array, ndarray, inf, atleast_2d, zeros, sum, median, where
from pytransit import sdss_g, sdss_r, sdss_i, sdss_z, RoadRunnerModel
from pytransit.contamination import Instrument, SMContamination
from pytransit.lpf.cntlpf import contaminate
from pytransit.lpf.tess.tgclpf import BaseTGCLPF
from pytransit.orbits import as_from_rhop, i_from_ba
from uncertainties import ufloat

sys.path.append('..')
from src.core import read_tess, read_m2, read_hipercam

import astropy.units as u

mj2kg = u.M_jup.to(u.kg)
ms2kg = u.M_sun.to(u.kg)
d2s = u.day.to(u.s)

# Stellar parameters from ALFOSC spectrum
# ---------------------------------------
star_teff = ufloat(3200,  160)
star_logg = ufloat( 5.0,  0.25)
star_z    = ufloat( 0.08,  0.08)
star_r    = ufloat(0.260, 0.010)
star_m    = ufloat(0.23, 0.02)

# Prior orbital parameters
# ------------------------
zero_epoch = ufloat(22458957.927167, 0.002598)
zero_epoch = ufloat(2459255.6937865, 0.005)
period = ufloat(2.326214, 0.000223)

# Photometry files
# ----------------
root = Path(__file__).parent.resolve()
m2_files = sorted((root / 'data' / 'muscat2').glob('*fits'))
tess_files = sorted((root / 'data' / 'tess').glob('*.fits'))

def read_external_data():
    times, fluxes, covs = [], [], []

    df = pd.read_csv('data/external/TIC8348911-01_02032021_TRAPPIST-North_z_measurements.xls', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1_n.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210305_LCO-McD-1m0_ip_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210326_LCO-CTIO-1m0_ip_10px_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210328_LCO-SSO-1m0_ip_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB_MOBS.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210401_TRAPPIST-South_I+z_Measurements.txt', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210416_LCO-CTIO-1m0_ip_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_23022021_TRAPPIST-North_z_measurements.xls', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC_8348911.01_26012021_TRAPPIST-North_I+z_measurements.txt', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.Rel_flux_T1.values.copy()
    c = df['Airmass FWHM dX dY'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    
    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-SSO-2m0_gp_Measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-SSO-2m0_rp_Measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-SSO-2m0_ip_Measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-SSO-2m0_zs_Measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)
    
    
    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-HAL-M3_gp_10px_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-HAL-M3_rp_15px_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-HAL-M3_ip_15px_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)

    df = pd.read_csv('data/external/TIC8348911-01_20210523_LCO-HAL-M3_zs_15px_measurements.tbl', delim_whitespace=True)
    t = df.BJD_TDB.values.copy()
    f = df.rel_flux_T1.values.copy()
    c = df['AIRMASS FWHM_Mean X(IJ)_T1 Y(IJ)_T1'.split()].values.copy()
    times.append(t); fluxes.append(f), covs.append(c)
    
    fluxes = [f / median(f) for f in fluxes]
    wns = [diff(f).std() / sqrt(2) for f in fluxes]
    pbs = 'z_s i i i i i z_s i g r i z_s g r i z_s'.split()
    #pbs = 'z_s i i i iz i z_s iz'.split()

    ins = 'TRAPPIST LCO LCO LCO TRAPPIST LCO TRAPPIST TRAPPIST M3 M3 M3 M3 M3 M3 M3 M3'.split()
    piis = list(arange(len(times)))
    return times, fluxes, pbs, wns, covs, ins, piis


# Define the log posterior functions
# ----------------------------------
# The `pytransit.lpf.tess.BaseTGCLPF` class can be used directly to model TESS photometry together with ground-based
# multicolour photometry with physical contamination on the latter. The only thing that needs to be implemented is the
# `BaseTGCLPF.read_data()` method that reads and sets up the data. However, we also use the `_post_initialisation`
# hook to set the parameter priors for the so that we don't need to remember to set these later.

class LPF(BaseTGCLPF):
    def __init__(self, name: str, use_ldtk: bool = False, use_opencl: bool = False, use_pdc: bool = True,
                 heavy_baseline: bool = True, downsample: Optional[float] = None, m2_passbands=('g', 'r', 'i', 'z_s')):
        self.use_pdc = use_pdc
        self.use_opencl = use_opencl
        self.heavy_baseline = heavy_baseline
        self.downsample = downsample
        self.m2_passbands = m2_passbands
        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=True)
        super().__init__(name, use_ldtk, tm=tm)

    def read_data(self):
        times_t, fluxes_t, pbs_t, wns_t, ins_t, piis_t = read_tess(tess_files, zero_epoch, period,
                                                                   baseline_duration_d=0.3,
                                                                   use_pdc=self.use_pdc)
        times_m2, fluxes_m2, pbs_m2, wns_m2, covs_m2, ins_m2, piis_m2 = read_m2(m2_files,
                                                                                downsample=self.downsample,
                                                                                passbands=self.m2_passbands)
        times_l, fluxes_l, pbs_l, wns_l, covs_l, ins_l, piis_l = read_external_data()
        times_h, fluxes_h, pbs_h, wns_h, covs_h, ins_h, piis_h = read_hipercam(['data/hipercam/toi-266.01-hipercam-210805.fits'])

        times = times_t + times_m2 + times_l + times_h
        fluxes = fluxes_t + fluxes_m2 + fluxes_l + fluxes_h
        pbs = pbs_t + pbs_m2 + pbs_l + pbs_h
        wns = wns_t + wns_m2 + wns_l + wns_h
        if self.heavy_baseline:
            covs = len(times_t) * [array([[]])] + covs_m2 + covs_l + covs_h
        else:
            covs = (len(times_t) + len(times_m2) + len(times_l))* [array([[]])] + len(times_h)

        pbnames = 'tess g r i z_s'.split()

        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])
        self.ins = ins_t + ins_m2 + ins_l + ins_h
        self.piis = piis_t + piis_m2 + piis_l + piis_h

        fluxes = [f / median(f) for f in fluxes]
        covs = [(c-c.mean(0)) / c.std(0) for c in covs]

        return times, fluxes, pbnames, pbs, wns, covs

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.add_prior(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
        self.add_prior(lambda pv: where(pv[:, 8] < pv[:, 5], 0, -inf))

    def _post_initialisation(self):
        if self.use_opencl:
            self.tm = self.tm.to_opencl()
        self.set_prior('tc', 'NP', zero_epoch.n, zero_epoch.s)
        self.set_prior('p', 'NP', period.n, period.s)
        self.set_prior('rho', 'UP', 20, 35)
        self.set_prior('k2_app', 'UP', 0.02 ** 2, 0.08 ** 2)
        self.set_prior('k2_true', 'UP', 0.02 ** 2, 0.95 ** 2)
        self.set_prior('k2_app_tess', 'UP', 0.02 ** 2, 0.08 ** 2)
        self.set_prior('teff_h', 'NP', star_teff.n, star_teff.s)
        self.set_prior('teff_c', 'UP', 2500, 12000)

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        zero_epoch = pvp[:,0] - self._tref
        period = pvp[:,1]
        smaxis = as_from_rhop(pvp[:, 2], period)
        inclination  = i_from_ba(pvp[:, 3], smaxis)
        radius_ratio = sqrt(pvp[:,5:6])
        ldc = pvp[:, self._sl_ld].reshape([-1, self.npb, 2])
        flux = self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        cnref = 1. - pvp[:, 4] / pvp[:, 5]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, 6], pvp[:, 7])
        return contaminate(flux, cnt, self.lcids, self.pbids)
