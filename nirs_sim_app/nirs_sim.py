import numpy as np
import pandas as pd
from pymcx import MCX

def simulate(spec, cw_analysis, fd_analysis, wavelength, modulation_frequency_mhz):
    cfg = spec['mcx']
    cfg.prop = spec['create_prop'](wavelength)
    run_count = spec.get('run_count', 1)
    if 'seeds' in spec:
        seeds = spec['seeds']
    else:
        seeds = np.random.randint(0xFFFF, size=run_count)
    results = []
    for seed in seeds:
        cfg.seed = int(seed)
        result = cfg.run(2)
        if result["exportdetected"].shape[1] >= cfg.maxdetphoton:
            raise Exception("Too many photons detected({}), check nphoton({}) and maxdetphoton({})".format(result["exportdetected"].shape, cfg.nphoton, cfg.maxdetphoton))
        results.append(result)
    detp = np.concatenate([result["exportdetected"] for result in results], axis=1)
    intesity_0 = cfg.nphoton * run_count * spec['areas']
    num_media = cfg.prop.shape[0] - 1
    dets = [detp[0] == adet for adet in spec['analysis_dets']]
    analysis = {'Photons': np.sum(dets, axis=1)}
    if cw_analysis:
        phiCW = np.array([np.sum(np.exp(-cfg.prop[1:, 0] @ detp[2:num_media + 2, d])) for d in dets])
        mcx_CW_Rd = phiCW / intesity_0
        analysis['Reflectance'] = mcx_CW_Rd
    if fd_analysis:
        FD_wavevector = 2 * np.pi * modulation_frequency_mhz * 1e6 * cfg.prop[1:, 3] / 2.998e11  # in units of mm
        phiFD = np.array([np.sum(np.exp((-cfg.prop[1:, 0] + FD_wavevector * 1j) @ detp[2:num_media + 2, d])) for d in dets])
        mcx_FD_Rd = phiFD / intesity_0
        mcx_FD_AC = np.abs(mcx_FD_Rd)
        mcx_FD_Phase = np.angle(mcx_FD_Rd)
        analysis['AC'] = mcx_FD_AC
        analysis['Phase'] = mcx_FD_Phase
    return pd.DataFrame(analysis, index=spec['rhos'])

