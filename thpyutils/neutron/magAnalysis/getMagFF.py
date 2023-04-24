from mantid.simpleapi import *
import numpy as np

def getMagFF(q, mag_ion):
    """
    Given a list of momentum transfer magnitudes, returns a list of equal size of the magnetic form factor
    squared or |F(Q)|^2. Wrapper for Mantid catalogued form factors.

    :param np.ndarry q: Array of momentum transfers
    :param str mag_ion: String containing ion and valence in MANTID notation, for example 'Ir4'.
        See Mantid documentation for allowed values.
    :return FF: List of values by which data must be divided.
    """
    if mag_ion is None:
        FFcorrection = np.ones(len(q))
    else:
        cw = CreateWorkspace(DataX=q, DataY=np.ones(len(q)))
        cw.getAxis(0).setUnit('MomentumTransfer')
        ws_corr = MagFormFactorCorrection(cw, IonName=mag_ion, FormFactorWorkspace='FF')
        FFcorrection = 1.0 / np.array(ws_corr[0].readY(0))

    return FFcorrection
