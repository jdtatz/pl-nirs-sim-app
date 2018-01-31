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
    results = []
    for seed in seeds:
        cfg.seed = int(seed)
        result = cfg.run(2)
        if result["exportdetected"].shape[1] >= cfg.maxdetphoton:
            raise Exception("Too many photons detected({}), check nphoton({}) and maxdetphoton({})".format(result["exportdetected"].shape, cfg.nphoton, cfg.maxdetphoton))
        results.append(result)
    detp = np.concatenate([result["exportdetected"] for result in results], axis=1)
    tof_domain = np.append(np.arange(cfg.tstart, cfg.tend, cfg.tstep), cfg.tend)
    c = 2.998e+11 # speed of light in mm / s
    detTOF = (cfg.prop[1:, 3] @ detp[2:(len(cfg.prop) + 1)]) / c
    tofBins = np.digitize(detTOF, tof_domain) - 1
    detBins = detp[0].astype(np.intc) - 1
    n1, n2 = len(cfg.detpos), len(tof_domain)
    photon_counts = np.histogram(detBins*n2+tofBins, np.arange(n1*n2+1))[0].reshape((n1, n2))
    # photon_counts = np.histogramdd((detBins, tofBins), (np.arange(n1+1), np.arange(n2+1)))[0].astype(np.intc)
    partialVec = np.exp(-cfg.prop[1:, 0] @ detp[2:(len(cfg.prop) + 1)])
    phiTD = np.zeros((n1, n2), np.float32)
    np.add.at(phiTD, (detBins, tofBins), partialVec)
    return {'Photons': photon_counts, 'Phi': phiTD, 'Seeds': seeds}
