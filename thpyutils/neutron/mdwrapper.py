from mantid.simpleapi import *
import numpy as np
from thpyutils.neutron.methods import bin1D
from thpyutils.neutron.methods import normalizeMDhisto_event
from thpyutils.neutron.methods import undo_normalizeMDhisto_event

import thpyutils.neutron.methods.mdutils as mdu

class MDwrapper:
    """
    Class for convenience to handle mantid MD objects.
    """

    def __init__(self, name, filename=None, material=None):
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
        self.instrument = None  # Optional description of instrument.
        if filename is not None:
            self.filename = filename
        else:
            self.filename = None
        self.material = material  # Associated material object.
        self.vmin=None # for plotting
        self.vmax=None # for plotting

    def powderNXSPEtoMDHisto(self, q_slice, e_slice, file_arr=None, mt_arr=False, self_shield=1.0,
                             eventnorm=True):
        """
        Takes file array and turns it into MDhisto workspace (powder measurements only).

        :param list,str file_arr: List of reduced NXSPE files to import, or a single file. Input to this function will default to
            the .filename attribute, but can be overwritten using this argument.
        :param list,str mt_arr: Optional list of background measurements to subtract directly.
        :param list q_slice: List in the form of [Qmin,Qmax,Qstep].
        :param list e_slice: List in the form of [Emin,Emax,Estep]
        :param float self_shield: Self shielding factor if MT array is specified for background subtraction.
        :param bool eventnorm: Flag to normalize by number of monitor events, defaults to True.
        """
        # First reformat the slice definitions to be friendly with BinMD

        Q_slice = '|Q|,' + str(q_slice[0]) + ',' + str(q_slice[1]) + ',' + str(round(np.abs(q_slice[1] - q_slice[0])
                                                                                     / q_slice[2]))
        E_slice = 'DeltaE,' + str(e_slice[0]) + ',' + str(e_slice[1]) + ',' + str(round(np.abs(e_slice[1] - e_slice[0])
                                                                                        / e_slice[2]))

        if (self.filename is not None) and file_arr is None:
            # By default, any input to this function in the file_arr argument will supercede the .filename attribute
            file_arr = self.filename

        if type(file_arr) == str:
            file_arr = [file_arr]
        elif type(file_arr) is not list:
            print("Error in specification of input file(s).")
            return 0
        if type(mt_arr) == str:
            mt_arr = [mt_arr]
        matrix_ws = LoadNXSPE(file_arr[0])
        if mt_arr is not False:
            matrix_ws_MT = LoadNXSPE(mt_arr[0])
            i = 1
            while len(mt_arr) > i:
                x = LoadNXSPE(mt_arr[i])
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
        if mt_arr is not False:
            md_MT = ConvertToMD(matrix_ws_MT, Qdimensions='|Q|')
            # Normalize to event.
        print(md_smpl)
        # Bin both
        cut2D_smpl = BinMD(md_smpl, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
        # Normalize to event
        if eventnorm:
            cut2D_smpl = normalizeMDhisto_event(cut2D_smpl)
            self.event_normalized = True
        if mt_arr is not False:
            cut2D_MT = BinMD(md_MT, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
            # normalize to event
            if eventnorm:
                cut2D_MT = normalizeMDhisto_event(cut2D_MT)
                self.event_normalized = True
            cut2D = cut2D_smpl - self_shield * cut2D_MT
        else:
            cut2D = cut2D_smpl
        CloneWorkspace(cut2D,OutputWorkspace=self.name+"_MD")
        self.mdhisto = mtd[self.name+'_MD']
        self.sampletype = 'powder'
        self.binned_events = True

    def colorplot(self,fig,ax,vmin,vmax,cmap,cbar=False):
        """

        :param fig: Input matplotlib figure object
        :param ax: Input matplotlib axes
        :param vmin: Colormap minimum
        :param vmax: Colormap maximum
        :param cmap: String of matplotlib colormap
        :param bool cbar: Flag for autogeneration of the colorbar by mantid.
        :return im: Mesh mappable object returned by pcolormesh
        """
        if self.sampletype=='powder':
            pass
        else:
            print('Color plots for single crystal measurements are not supported yet. ')
            return 0
        cut2D = self.mdhisto
        #Use the built in MANTID plotting tools, these assume that the measurement is NOT yet normalized to events.
        cut2Dplt = cut2D.clone()
        cut2Dplt = undo_normalizeMDhisto_event(cut2Dplt)
        im = ax.pcolormesh(cut2Dplt,vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=10)
        ax.set_ylabel(r'$\hbar\omega$ (meV)', fontsize=10)
        return im