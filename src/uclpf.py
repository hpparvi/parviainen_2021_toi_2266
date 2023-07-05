from typing import Optional

import pandas as pd
from numpy import arange, concatenate, atleast_2d, ndarray, floor, array, median, sqrt
from pytransit import BaseLPF, RoadRunnerModel, LinearModelBaseline
from pytransit.orbits import as_from_rhop, i_from_ba

from src.io import read_tess_data, read_m2_data, read_lco_data, read_hipercam_data
from toi_2266 import zero_epoch, period


class UCLPF(BaseLPF):
    def __init__(self, parallelize: bool = True):
        self.use_pdc = True
        self.heavy_baseline = True
        self.downsample = None
        self.m2_passbands = ('r', 'i', 'z_s')

        times, fluxes, pbnames, pbs, wns, covs = self.read_data()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = arange(len(times))
        tref = floor(concatenate(times).min())

        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=parallelize)
        super().__init__("toi-2266.01-joint-uncontaminated", "tess g r i z_s".split(), times, fluxes, pbids=pbids, wnids=wnids,
                         covariates=covs, tm=tm, tref=tref)
        self._add_baseline_model(LinearModelBaseline(self))

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
        pbs = pbs_t + pbs_m2 + pbs_l + pbs_h
        wns = wns_t + wns_m2 + wns_l + wns_h
        if self.heavy_baseline:
            covs = len(times_t) * [array([[]])] + covs_m2 + covs_l + covs_h
        else:
            covs = (len(times_t) + len(times_m2) + len(times_l)) * [array([[]])] + len(times_h)

        pbnames = 'tess g r i z_s'.split()

        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])
        self.ins = ins_t + ins_m2 + ins_l + ins_h
        self.piis = piis_t + piis_m2 + piis_l + piis_h

        fluxes = [f / median(f) for f in fluxes]
        covs = [(c - c.mean(0)) / c.std(0) for c in covs]

        return times, fluxes, pbnames, pbs, wns, covs

    def _post_initialisation(self):
        self.set_prior('tc', 'NP', zero_epoch.n, zero_epoch.s)
        self.set_prior('p', 'NP', period.n, period.s)
        self.set_prior('rho', 'UP', 10, 35)
        self.set_prior('k2', 'UP', 0.0005, 0.006)
        self.set_prior('q1_tess', 'NP', 0.78, 0.02)
        self.set_prior('q2_tess', 'NP', 0.77, 0.02)
        self.set_prior('q1_g', 'NP', 0.64, 0.02)
        self.set_prior('q2_g', 'NP', 0.64, 0.02)
        self.set_prior('q1_r', 'NP', 0.65, 0.02)
        self.set_prior('q2_r', 'NP', 0.59, 0.02)
        self.set_prior('q1_i', 'NP', 0.75, 0.02)
        self.set_prior('q2_i', 'NP', 0.72, 0.02)
        self.set_prior('q1_z_s', 'NP', 0.79, 0.02)
        self.set_prior('q2_z_s', 'NP', 0.78, 0.02)

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        ldc = pv[:, self._sl_ld].reshape([-1, self.npb, 2])
        zero_epoch = pv[:, 0] - self._tref
        period = pv[:, 1]
        smaxis = as_from_rhop(pv[:, 2], period)
        inclination = i_from_ba(pv[:, 3], smaxis)
        radius_ratio = sqrt(pv[:, 4:5])
        return self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)