import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('salamandrasnn')[0] + 'salamandrasnn'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)
from network_experiments.default_parameters.salamander.walking.walking_default import *

###############################################################################
## CONSTANTS ##################################################################
###############################################################################
MODEL_NAME = '4limb_1dof_unweighted'
MODNAME    = MODELS_OPENLOOP[MODEL_NAME][0]
PARSNAME   = MODELS_OPENLOOP[MODEL_NAME][1]

###############################################################################
## PARAMETERS #################################################################
###############################################################################
def get_default_parameters():
    ''' Get the default parameters for the analysis '''

    # Process parameters
    params_process = PARAMS_SIMULATION

    return {
        'modname'              : MODELS_OPENLOOP[MODEL_NAME][0],
        'parsname'             : MODELS_OPENLOOP[MODEL_NAME][1],
        'params_process'       : params_process,
    }