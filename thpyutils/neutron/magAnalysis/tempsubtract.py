import numpy as np
from mantid.simpleapi import *
from thpyutils.neutron.methods import mdutils as mdu

def tempsubtractMD(lowTobj, highTobj):
    """
    Function to subtract a high temperature dataset from a lower one after scaling by a Bose-Einstein
    population factor
    :param lowTobj: mdwrapper object for the low temperature data
    :param highTobj: mdwrapper object for the high temperature dta
    :return magmdhisto: mdhistoworkspace of the subtracted data
    """

    tLow = lowTobj.temperature
    tHigh = highTobj.temperature
    highTMD = highTobj.mdhisto
    print(highTMD)
    lowTMD = lowTobj.mdhisto
    highT_cut2D_T = CloneWorkspace(highTMD,OutputWorkspace='tmpMD_high')
    lowT_cut2D_T = CloneWorkspace(lowTMD,OutputWorkspace='tmpMD_low')

    dims = lowT_cut2D_T.getNonIntegratedDimensions()

    energies = mdu.dim2array(dims[1])

    kb = 8.617e-2
    bose_factor_lowT = (1 - np.exp(-energies / (kb * tLow)))
    bose_factor_highT = (1 - np.exp(-energies / (kb * tHigh)))
    # Only makes sense for positive transfer
    bose_factor_lowT[np.where(energies < 0)] = 0
    bose_factor_highT[np.where(energies < 0)] = 0
    highT_Intensity = np.copy(highT_cut2D_T.getSignalArray())
    highT_err = np.sqrt(np.copy(highT_cut2D_T.getErrorSquaredArray()))
    bose_factor = bose_factor_highT / bose_factor_lowT
    highT_Intensity_corrected = bose_factor * highT_Intensity
    highT_err_corrected = bose_factor * highT_err
    highT_Intensity_corrected[np.where(highT_Intensity_corrected == 0)] = 0
    highT_err_corrected[np.where(highT_err_corrected == 0)] = 0
    highT_Intensity_corrected[np.isnan(highT_Intensity_corrected)] = 0
    highT_err_corrected[np.isnan(highT_err_corrected)] = 0

    highT_cut2D_T.setSignalArray(highT_Intensity_corrected)
    highT_cut2D_T.setErrorSquaredArray(highT_err_corrected ** 2)

    lowT_cut2D_intensity = np.copy(lowT_cut2D_T.getSignalArray())
    lowT_cut2D_err = np.sqrt(np.copy(lowT_cut2D_T.getErrorSquaredArray()))

    mag_intensity = lowT_cut2D_intensity - highT_Intensity_corrected
    mag_err = np.sqrt(lowT_cut2D_err ** 2 + highT_err_corrected ** 2)

    cut2D_mag_tempsub = CloneWorkspace(lowT_cut2D_T,OutputWorkspace=lowTobj.name+'_highTSub')
    cut2D_mag_tempsub.setSignalArray(mag_intensity)
    cut2D_mag_tempsub.setErrorSquaredArray(mag_err ** 2)
    return cut2D_mag_tempsub
