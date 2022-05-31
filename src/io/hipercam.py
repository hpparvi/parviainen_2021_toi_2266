from pathlib import Path

from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table

from numpy import arange
from pytransit.utils.downsample import downsample_time_1d, downsample_time_2d

hipercam_path = (Path(__file__).parent.parent.parent / 'data/light_curves/hipercam/').resolve()
hipercam_files = sorted(hipercam_path.glob('*.fits'))

def read_hipercam_data(downsample=None, passbands=('g', 'r', 'i', 'z')):
    files = hipercam_files
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for inight, f in enumerate(files):
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for ipb in range(npb):
                hdu = hdul[1 + ipb]
                pb = hdu.header['filter']
                if pb in passbands:
                    fobs = hdu.data['flux'].astype('d').copy()
                    fmod = hdu.data['model'].astype('d').copy()
                    time = hdu.data['time_bjd'].astype('d').copy()
                    mask = ~sigma_clip(fobs-fmod, sigma=5).mask

                    wns.append(hdu.header['wn'])
                    if pb == 'z':
                        pb = 'z_s'
                    pbs.append(pb)

                    if downsample is None:
                        times.append(time[mask])
                        fluxes.append(fobs[mask])
                        covs.append(Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:])
                    else:
                        cov = Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:]
                        tb, fb, _ = downsample_time_1d(time[mask], fobs[mask], downsample / 24 / 60)
                        _,  cb, _ = downsample_time_2d(time[mask], cov, downsample / 24 / 60)
                        times.append(tb)
                        fluxes.append(fb)
                        covs.append(cb)
    ins = len(times)*["HIP"]
    piis = list(arange(len(times)))
    return times, fluxes, pbs, wns, covs, ins, piis
