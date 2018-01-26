#                                                            _
# nirs_sim_app ds app
#
# (c) 2016 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import bz2
import pickle

import numpy as np
from pymcx import MCX
from nirs_sim import simulate

# import the Chris app superclass
from chrisapp.base import ChrisApp



class Nirs_sim_app(ChrisApp):
    """
    An app to run nirs mcx simulations..
    """
    AUTHORS         = 'FNNDSC (jacob.tatz@childrens.harvard.edu)'
    SELFPATH        = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC        = os.path.basename(__file__)
    EXECSHELL       = 'python3'
    TITLE           = 'Nirs Sims App'
    CATEGORY        = ''
    TYPE            = 'ds'
    DESCRIPTION     = 'An app to run nirs mcx simulations.'
    DOCUMENTATION   = 'http://wiki'
    VERSION         = '0.1'
    LICENSE         = 'Opensource (MIT)'

    # Fill out this with key-value output descriptive info (such as an output file path
    # relative to the output dir) that you want to save to the output meta file when
    # called with the --saveoutputmeta flag
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        """
        self.add_argument(
            '--spec',
            dest        = 'spec_file',
            type        = str,
            optional    = False,
            help        = 'MCX Specification file to run'
        )

        self.add_argument(
            '--wavelength',
            dest        = 'wavelength',
            type        = float,
            optional    = False,
            help        = 'Wavelength to simulate'
        )

        self.add_argument(
            '--cw',
            dest        = 'cw_analysis',
            type        = bool,
            default     = True,
            optional    = True,
            help        = 'Analyze results for CWNIRS'
        )

        self.add_argument(
            '--fd',
            dest        = 'fd_analysis',
            type        = bool,
            default     = True,
            optional    = True,
            help        = 'Analyze results for FDNIRS'
        )

        self.add_argument(
            '--modulation-frequncy',
            dest        = 'modulation_frequency_mhz',
            type        = float,
            default     = 110,
            optional    = True,
            help        = 'Modulation Frequency in MHz to analyze for FDNIRS'
        )


    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        # Load simulation params
        with bz2.open(os.path.join(options.inputdir, options.spec_file)) as f_in:
            spec = pickle.loads(f_in.read())
        # Run simulation
        results = simulate(spec, options.cw_analysis, options.fd_analysis, options.wavelength, options.modulation_frequency_mhz)
        # Save Results
        out_name = "results_{}_{}.bz2".format(spec['uuid'], options.wavelength)
        with bz2.open(os.path.join(options.outputdir, out_name), 'w') as arc:
            arc.write(pickle.dumps(results))

# ENTRYPOINT
if __name__ == "__main__":
    app = Nirs_sim_app()
    app.launch()
