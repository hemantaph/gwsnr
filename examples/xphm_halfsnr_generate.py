import numpy as np
from gwsnr import GWSNR as snr


snr_ = snr(npool=int(4), mtot_min=2., mtot_max=439.6, nsamples_mtot=200,\
            nsamples_mass_ratio=500,\
            sampling_frequency=4096.,\
            waveform_approximant = 'IMRPhenomXPHM', minimum_frequency = 20.,\
            snr_type = 'interpolation',\
            waveform_inspiral_must_be_above_fmin=False)