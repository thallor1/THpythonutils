from mantid.simpleapi import *
import numpy as np
from neutron.methods import bin1D
from neutron.methods import normalizeMDhisto_event


class MDwrapper:
    """
    Class for convenience to handle mantid MD objects.
    """

    def __init__(self, name, filename=None):
        # The object itself is initialized to be completely empty.
        self.mdhisto = None  # Actual MANTID mdhisto object.
        self.name = name  # Mandatory string when initializing the object.
        self.sampletype = 'powder'  # Allowed to be 'powder' or 'crystal'
        self.mag_ion = None  # Magnetic ion for form factor calculations.
        self.samplemass = None
        self.binned_events = True  # Defines if this is an MD or MDHisto object
        self.temperature = 300.0  # Measurement temperature in Kelvin
        self.field = 0.0  # Measurement field in tesla.
        self.event_normalized = False  # Flag if the measurement has already been normalized by events.
        self.instrument = None # Optional description of instrument.
        if filename is not None:
            self.filename = filename
        else:
            self.filename = None
        self.material = None # Associated material object.

    def powderNXSPEtoMDHisto(self, file_arr, MT_arr=False, Q_slice='[0,3,0.1]', E_slice='[-10,10,0.1]', self_shield=1.0,
                             numEvNorm=True):
        """
        Takes file array and turns it into MDhisto workspace (powder measurements only).

        :param file_arr: List of reduced NXSPE files to import.
        :param MT_arr: Optional list of background measurements to subtract directly.
        :param Q_slice: List in the form of '[Qmin,Qmax,Qstep]', formatted as a string.
        :param E_slice: List in the form of '[Emin,Emax,Estep]', formatted as a string.
        :param self_shield: Self sheilding factor if MT array is specified for background subtraction.
        :param numEvNorm: Flag to normalize by number of monitor events, defaults to True.
        """

        if type(file_arr) == str:
            file_arr = [file_arr]
        elif type(file_arr) is not list:
            print("Error in specification of input file(s).")
            return 0
        if type(MT_arr) == str:
            MT_arr = [MT_arr]
        matrix_ws = LoadNXSPE(file_arr[0])
        if MT_arr is not False:
            matrix_ws_MT = LoadNXSPE(MT_arr[0])
            i = 1
            while len(MT_arr) > i:
                x = LoadNXSPE(MT_arr[i])
                matrix_ws_MT = matrix_ws_MT + x
                i = i + 1
            matrix_ws_MT = matrix_ws_MT / i
        i = 1
        while len(file_arr) > i:
            x = LoadNXSPE(file_arr[i])
            matrix_ws = matrix_ws + x
            i = i + 1
        matrix_ws = matrix_ws / i

        # Convert to MD
        md_smpl = ConvertToMD(matrix_ws, Qdimensions='|Q|')
        if MT_arr is not False:
            md_MT = ConvertToMD(matrix_ws_MT, Qdimensions='|Q|')
            # Normalize to event.

        # Bin both
        cut2D_smpl = BinMD(md_smpl, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
        # Normalize to event
        if numEvNorm:
            cut2D_smpl = normalizeMDhisto_event(cut2D_smpl)
            self.event_normalized = True
        if MT_arr is not False:
            cut2D_MT = BinMD(md_MT, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
            # normalize to event
            if numEvNorm:
                cut2D_MT = normalizeMDhisto_event(cut2D_MT)
                self.event_normalized = True
            cut2D = cut2D_smpl - self_shield * cut2D_MT
        else:
            cut2D = cut2D_smpl
        self.mdhisto = cut2D.clone()
        self.sampletype = 'powder'
        self.binned_events = True

