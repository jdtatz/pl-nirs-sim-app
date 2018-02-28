import numpy as np
import numba as nb
from pymcx import MCX

def create_prop(spec, wavelen):
    concentrations = spec['concentrations']
    a = spec['scatter_a']
    b = spec['scatter_b']
    g = spec.get('g', 0.9)
    ind = np.where(spec['waves'] == wavelen)[0][0]
    extinction_coeffs = spec['lut'][ind] / 10
    media = np.empty((len(concentrations) + 1, 4), np.float32)
    media[0] = [0, 0, 1, spec.get('n_external', 1)]
    media[1:, 0] = concentrations @ extinction_coeffs  # mua
    media[1:, 1] = (a * wavelen ** (-b)) / (1-g)  # mus
    media[1:, 2] = g
    media[1:, 3] = spec.get('n', 1.37)
    return media


@nb.jit(nopython=True, nogil=True, parallel=False)
def analysis(detp, prop, tof_domain, tau, wavelength, BFi, freq, ndet, ntof, nmedia, pcounts, paths, phiTD, phiFD, g1_top, phiDist):
    c = 2.998e+11 # speed of light in mm / s
    detBins = detp[0].astype(np.intc) - 1
    tofBins = np.minimum(np.digitize(prop[1:, 3] @ detp[2:(2+nmedia)], c * tof_domain), ntof) - 1
    distBins = np.minimum(np.digitize(prop[1:, 3] * detp[2:(2+nmedia)].T, c * tof_domain), ntof) - 1
    path = -prop[1:, 0] @ detp[2:(2+nmedia)]
    phis = np.exp(path)
    fds = np.exp((-prop[1:, 0] + 2j * np.pi * freq * prop[1:, 3] / c).astype(np.complex64) @ detp[2:(2+nmedia)].astype(np.complex64))
    prep = (-2*(2*np.pi*prop[1:, 3]/(wavelength*1e-6))**2*BFi).astype(np.float32) @ detp[(2+nmedia):(2+2*nmedia)]
    for i in range(len(detBins)):
        pcounts[detBins[i], tofBins[i]] += 1
        paths[detBins[i], tofBins[i]] += detp[2:(2+nmedia), i]
        phiTD[detBins[i], tofBins[i]] += phis[i]
        phiFD[detBins[i]] += fds[i]
        for l in range(nmedia):
            phiDist[detBins[i], distBins[i, l], l] += phis[i]
        for j in range(len(tau)):
            g1_top[detBins[i], j] += np.exp(prep[i] * tau[j] + path[i])


def simulate(spec, wavelength):
    cfg = spec['mcx']
    cfg.ismomentum = True
    cfg.prop = create_prop(spec, wavelength)
    run_count = spec.get('run_count', 1)
    seeds = np.asarray(spec.get('seeds', np.random.randint(0xFFFF, size=run_count)))
    tof_domain = spec.get('tof_domain', np.append(np.arange(cfg.tstart, cfg.tend, cfg.tstep), cfg.tend))
    tau = spec.get('tau', np.logspace(-8, -2))
    BFi = spec['BFi']
    freq = spec.get('frequency', 110e6)
    ndet, ntof, nmedia = len(cfg.detpos), len(tof_domain) - 1, len(cfg.prop) - 1
    phiTD = np.zeros((ndet, ntof), np.float64)
    phiFD = np.zeros(ndet, np.complex128)
    paths = np.zeros((ndet, ntof, nmedia), np.float64)
    pcounts = np.zeros((ndet, ntof), np.int64)
    g1_top = np.zeros((ndet, len(tau)), np.float64)
    phiDist = np.zeros((ndet, ntof, nmedia), np.float64)
    fslice = 0
    for seed in seeds:
        cfg.seed = int(seed)
        result = cfg.run(2)
        detp = result["detphoton"]
        if detp.shape[1] >= cfg.maxdetphoton:
            raise Exception("Too many photons detected: {}".format(detp.shape[1]))
        analysis(detp, cfg.prop, tof_domain, tau, wavelength, BFi, freq, ndet, ntof, nmedia, pcounts, paths, phiTD, phiFD, g1_top, phiDist)
        fslice += result["fluence"][spec['slice']]
        del detp
        del result
    fslice /= run_count
    paths /= pcounts[:, :, np.newaxis]
    g1 = g1_top / np.sum(phiTD, axis=1)[:, np.newaxis]
    phiDist /= np.sum(phiTD, axis=1)[:, np.newaxis, np.newaxis]
    return {'Photons': pcounts, 'Paths': paths, 'PhiTD': phiTD, 'PhiFD': phiFD, 'PhiDist': phiDist, 'Seeds': seeds, 'Slice': fslice, 'g1': g1}
