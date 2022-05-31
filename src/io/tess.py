from copy import copy
from pathlib import Path
from typing import List

from astropy.table import Table
from numpy import zeros, diff, concatenate, sqrt, arange
from pytransit.utils.keplerlc import KeplerLC
from uncertainties import nominal_value

tess_path = (Path(__file__).parent.parent.parent / 'data/light_curves/tess/').resolve()
tess_files = sorted(tess_path.glob('*.fits'))

def read_tess_data(zero_epoch: float, period: float, use_pdc: bool = False,
              transit_duration_d: float = 0.1, baseline_duration_d: float = 0.3):
    dfiles = tess_files
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    times, fluxes, ins, piis = [], [], [], []
    for dfile in dfiles:
        tb = Table.read(dfile)
        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
        lc = KeplerLC(df.TIME.values + bjdrefi, df[fcol].values, zeros(df.shape[0]),
                      nominal_value(zero_epoch), nominal_value(period), transit_duration_d, baseline_duration_d)
        times.extend(copy(lc.time_per_transit))
        cfluxes = copy(lc.normalized_flux_per_transit)
        if use_pdc:
            contamination = 1 - tb.meta['CROWDSAP']
            cfluxes = [contamination + (1 - contamination) * f for f in cfluxes]
        fluxes.extend(cfluxes)

    ins = len(times) * ["TESS"]
    piis = list(arange(len(times)))
    return times, fluxes, len(times) * ['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)], ins, piis