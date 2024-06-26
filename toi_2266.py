import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from numpy import diff, sqrt, arange, array, ndarray, inf, atleast_2d, zeros, sum, median, where, repeat, unique
from pytransit import sdss_g, sdss_r, sdss_i, sdss_z, RoadRunnerModel
from pytransit.contamination import Instrument, SMContamination
from pytransit.lpf.cntlpf import contaminate
from pytransit.lpf.tess.tgclpf import BaseTGCLPF
from pytransit.orbits import as_from_rhop, i_from_ba, epoch
from pytransit.param import GParameter, UniformPrior as UP, NormalPrior as NP
from uncertainties import ufloat

sys.path.append('..')
from src.io import read_tess_data, read_m2_data, read_hipercam_data, read_lco_data
from src.core import zero_epoch, period, star_teff


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
        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=False)
        super().__init__(name, use_ldtk, tm=tm)
        self.result_dir = Path('results')

    def read_data(self):
        times_t, fluxes_t, pbs_t, wns_t, ins_t, piis_t = read_tess_data(zero_epoch, period,
                                                                   baseline_duration_d=0.3,
                                                                   use_pdc=self.use_pdc)
        times_m2, fluxes_m2, pbs_m2, wns_m2, covs_m2, ins_m2, piis_m2 = read_m2_data(downsample=self.downsample,
                                                                                     passbands=self.m2_passbands)
        times_l, fluxes_l, pbs_l, wns_l, covs_l, ins_l, piis_l = read_lco_data()
        times_h, fluxes_h, pbs_h, wns_h, covs_h, ins_h, piis_h = read_hipercam_data()

        times = times_t + times_m2 + times_l + times_h
        fluxes = fluxes_t + fluxes_m2 + fluxes_l + fluxes_h

        self.epochs = epoch(array([median(a) for a in times]), zero_epoch.n, period.n)
        self.epids = pd.Categorical(self.epochs).codes
        self.nepochs = self.epids.max() + 1

        pbs = pbs_t + pbs_m2 + pbs_l + pbs_h
        wns = wns_t + wns_m2 + wns_l + wns_h
        covs = len(times_t) * [array([[]])] + covs_m2 + covs_l + covs_h
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
        self.add_prior(lambda pv: where(pv[:, self._i_k2a] < pv[:, self._i_k2t], 0, -inf))
        self.add_prior(lambda pv: where(pv[:, self._i_k2at] < pv[:, self._i_k2t], 0, -inf))

    def _post_initialisation(self):

        self._i_k2a = self.ps.find_pid('k2_app')
        self._i_k2t = self.ps.find_pid('k2_true')
        self._i_k2at = self.ps.find_pid('k2_app_tess')
        self._i_teffh = self.ps.find_pid('teff_h')
        self._i_teffc = self.ps.find_pid('teff_c')

        if self.use_opencl:
            self.tm = self.tm.to_opencl()

        self.set_prior('tc', 'NP', zero_epoch.n, 5*zero_epoch.s)
        self.set_prior('p', 'NP', period.n, period.s)
        self.set_prior('rho', 'UP', 5, 35)
        self.set_prior('k2_app', 'UP', 0.02 ** 2, 0.08 ** 2)
        self.set_prior('k2_true', 'UP', 0.02 ** 2, 0.95 ** 2)
        self.set_prior('k2_app_tess', 'UP', 0.02 ** 2, 0.08 ** 2)
        self.set_prior('teff_h', 'NP', star_teff.n, star_teff.s)
        self.set_prior('teff_c', 'UP', 2500, 12000)

        self.set_prior('q1_tess', 'NP', 0.78, 0.008)
        self.set_prior('q2_tess', 'NP', 0.69, 0.124)
        self.set_prior('q1_g', 'NP', 0.64, 0.014)
        self.set_prior('q2_g', 'NP', 0.61, 0.070)
        self.set_prior('q1_r', 'NP', 0.65, 0.015)
        self.set_prior('q2_r', 'NP', 0.56, 0.079)
        self.set_prior('q1_i', 'NP', 0.74, 0.012)
        self.set_prior('q2_i', 'NP', 0.68, 0.131)
        self.set_prior('q1_z_s', 'NP', 0.79, 0.011)
        self.set_prior('q2_z_s', 'NP', 0.71, 0.155)

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
        zero_epoch = pvp[:, 0] - self._tref
        period = pvp[:, 1]
        smaxis = as_from_rhop(pvp[:, 2], period)
        inclination  = i_from_ba(pvp[:, 3], smaxis)
        radius_ratio = sqrt(pvp[:, self._i_k2t : self._i_k2t+1])
        ldc = pvp[:, self._sl_ld].reshape([-1, self.npb, 2])
        flux = self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)
        cnt[:, 0] = 1 - pvp[:, self._i_k2at] / pvp[:, self._i_k2t]
        cnref = 1. - pvp[:, self._i_k2a] / pvp[:, self._i_k2t]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, self._i_teffh], pvp[:, self._i_teffc])
        return contaminate(flux, cnt, self.lcids, self.pbids)
