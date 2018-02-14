import numpy as np
from pymcx import MCX

def create_prop(spec, wavelen):
    concentrations = spec['concentrations']
    a = spec['scatter_a']
    b = spec['scatter_b']
    g = spec.get('g', 0.9)
    ind = np.where(spec['waves'] == wavelen)[0][0]
    extinction_coeffs = spec['lut'][ind] / 10
    media = np.empty((len(concentrations) + 1, 4), np.float64)
    media[0] = [0, 0, 1, spec.get('n_external', 1)]
    media[1:, 0] = concentrations @ extinction_coeffs  # mua
    media[1:, 1] = (a * wavelen ** (-b)) / (1-g)  # mus
    media[1:, 2] = g
    media[1:, 3] = spec.get('n', 1.37)
    return media


def simulate(spec, wavelength):
    cfg = spec['mcx']
    cfg.prop = create_prop(spec, wavelength)
    run_count = spec.get('run_count', 1)
    seeds = np.asarray(spec.get('seeds', np.random.randint(0xFFFF, size=run_count)))
    tof_domain = np.append(np.arange(cfg.tstart, cfg.tend, cfg.tstep), cfg.tend)
    c = 2.998e+11 # speed of light in mm / s
    n1, n2 = len(cfg.detpos), len(tof_domain) - 1
    phiTD = np.zeros((n1, n2), np.float64)
    photon_counts = 0
    fslice = 0
    for seed in seeds:
        cfg.seed = int(seed)
        result = cfg.run(2)
        detp = result["exportdetected"]
        if detp.shape[1] >= cfg.maxdetphoton:
            raise Exception("Too many photons detected: {}".format(detp.shape[1]))
        detBins = detp[0].astype(np.intc) - 1
        tofBins = np.digitize(cfg.prop[1:, 3] @ detp[2:(len(cfg.prop) + 1)] / c, tof_domain) - 1
        tofBins[tofBins == n2] -= 1
        np.add.at(phiTD, (detBins, tofBins), np.exp(-cfg.prop[1:, 0] @ detp[2:(len(cfg.prop) + 1)]))
        photon_counts += np.bincount(detBins*n2+tofBins, minlength=n1*n2).reshape((n1, n2))
        fslice += result["exportfield"][spec['slice']]
        del tofBins
        del detBins
        del detp
        del result
    fslice /= run_count
    return {'Photons': photon_counts, 'Phi': phiTD, 'Seeds': seeds, 'Slice': fslice}
