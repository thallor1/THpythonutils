from mantid.simpleapi import *
import numpy as np
from thpyutils.neutron.methods import bin1D
from thpyutils.neutron.methods import normalizeMDhisto_event
from thpyutils.neutron.methods import undo_normalizeMDhisto_event
from thpyutils.neutron.magAnalysis import tempsubtractMD

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
        self.samplemass = None # Sample mass
        self.Ei = None# Incident energy
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
        self.vmin = None  # for plotting
        self.vmax = None  # for plotting
        self.Qbins = None # Stored Q-bins 
        self.Ebins = None # Stored energy bins

    def annularAbsorbNXSPE(self, mat_string, outer_r, inner_r, samp_thick, samp_h,
                             num_density, eventnorm=True):
        """
        Function for the correction of annular absorption. This requires reloading the file itself.

        :param mat_string: Material string in the format specified in Mantid documentation.
        :param outer_r: Outer annulus radius in cm
        :param inner_r: Inner annulus radius in cm
        :param samp_thick: thickness of sample in cm
        :param samp_h: height of sample in cm
        :param num_density: Number density of sample in f.u. / Ang^3
        :param eventnorm: Specifies if intensities should be normalized to monitor at the end.
        :return: 
        """
        # First backup the pre-absorption corrected mdhistoworkspace.
        CloneWorkspace(self.mdhisto,OutputWorkspace=self.name+'_noabs')
        Ei=self.Ei
        output_ws_name = self.name + "_absorbcorr"
        f_list = self.filename
        Q_slice = self.Qbins
        E_slice = self.Ebins
        # First load the data
        Load(Filename=f_list, \
             OutputWorkspace=output_ws_name)
        load_ws = mtd[output_ws_name]
        merged_ws = MergeRuns(load_ws)
        # Convert to wavelength
        wavelength_ws_name = output_ws_name + '_wavelength'
        ConvertUnits(InputWorkspace=merged_ws, \
                     OutputWorkspace=wavelength_ws_name, Target='Wavelength', EMode='Direct', EFixed=Ei)
        # Run Absorption Utility
        abs_ws_name = output_ws_name + '_ann_abs'
        wavelengthws = mtd[wavelength_ws_name]
        factors = AnnularRingAbsorption(InputWorkspace=wavelengthws, \
                              OutputWorkspace=abs_ws_name, CanOuterRadius=outer_r, CanInnerRadius=inner_r, \
                              SampleHeight=samp_h, SampleThickness=samp_thick, SampleChemicalFormula=mat_string, \
                              SampleNumberDensity=num_density)
        wavelengthws_corr = wavelengthws/factors

        # Convert back to Q
        abs_meV_ws = output_ws_name + '_ann_abs_meV'
        ConvertUnits(wavelengthws_corr, OutputWorkspace=abs_meV_ws, Target='DeltaE', Efixed=Ei,
                     Emode='Direct')
        working_ws = mtd[abs_meV_ws]

        # Convert to MD
        ws_corrected = ConvertToMD(working_ws, Qdimensions='|Q|')
        # Bin according to specified Q, E spacing
        outMD = BinMD(ws_corrected, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
        # Normalize by num events
        nevents = outMD.getNumEventsArray()
        if eventnorm is False:
            self.event_normalized=False
            pass
        else:
            I = np.copy(outMD.getSignalArray())
            Err = np.sqrt(np.copy(outMD.getErrorSquaredArray()))
            I /= nevents
            Err /= nevents
            outMD.setSignalArray(I)
            outMD.setErrorSquaredArray(Err ** 2)
            self.event_normalized=True
        outMD = outMD.clone()
        self.mdhisto=outMD
    
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
        self.Qbins = Q_slice
        self.Ebins = E_slice
        if (self.filename is not None) and file_arr is None:
            # By default, any input to this function in the file_arr argument will supercede the .filename attribute
            file_arr = self.filename

        if type(file_arr) == str:
            file_arr = [file_arr]
        elif type(file_arr) is not list:
            print("Error in specification of input file(s).")
            return
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
        CloneWorkspace(cut2D, OutputWorkspace=self.name + "_MD")
        self.mdhisto = mtd[self.name + '_MD']
        self.sampletype = 'powder'
        self.binned_events = True

    def colorplot(self, fig, ax, vmin, vmax, cmap, cbar=False):
        """
        Creates a simple colorplot of an MDHistoworkspace.

        :param fig: Input matplotlib figure object
        :param ax: Input matplotlib axes
        :param vmin: Colormap minimum
        :param vmax: Colormap maximum
        :param cmap: String of matplotlib colormap
        :param bool cbar: Flag for autogeneration of the colorbar by mantid.
        :return im: Mesh mappable object returned by pcolormesh
        """
        if self.sampletype == 'powder':
            pass
        else:
            print('Color plots for single crystal measurements are not supported yet. ')
            return 0
        cut2D = self.mdhisto
        # Use the built in MANTID plotting tools, these assume that the measurement is NOT yet normalized to events.
        cut2Dplt = cut2D.clone()
        cut2Dplt = undo_normalizeMDhisto_event(cut2Dplt)
        im = ax.pcolormesh(cut2Dplt, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=10)
        ax.set_ylabel(r'$\hbar\omega$ (meV)', fontsize=10)
        return im

    def powdercut(self, qbins, ebins):
        """
        Function to take cuts of already binned MDHistoworkspaces.

        :param qbins: Binning parameters in the momentum transfer direction in the format of [Qmin, Qmax, step]
        :param ebins: Binning parameters in the energy transfer direction in the format of [Qmin, Qmax, step]
        :return X, I, Err: Numpy arrays giving the coordinate along the cut direction, the intensities, and errors.
        """
        # Clean the intensities.
        cut2D = self.mdhisto
        workspace_cut1D = cut2D.clone()
        intensities = np.copy(workspace_cut1D.getSignalArray())
        errors = np.sqrt(np.copy(workspace_cut1D.getErrorSquaredArray() * 1.))
        errors[np.isnan(intensities)] = 1e30
        intensities[np.isnan(intensities)] = 0
        dims = workspace_cut1D.getNonIntegratedDimensions()
        q = mdu.dim2array(dims[0])
        e = mdu.dim2array(dims[1])
        if len(qbins) == 3:
            # First limit range in E
            e_slice = intensities[:,
                      np.intersect1d(np.where(e >= ebins[0]), np.where(e <= ebins[1]))]
            slice_errs = errors[:,
                         np.intersect1d(np.where(e >= ebins[0]), np.where(e <= ebins[1]))]
            # Integrate over E for all values of Q
            integrated_intensities = []
            integrated_errs = []
            for i in range(len(e_slice[:, 0])):
                q_cut_vals = e_slice[i]
                q_cut_err = slice_errs[i]

                q_cut_err = q_cut_err[np.intersect1d(np.where(q_cut_vals != 0)[0], np.where(~np.isnan(q_cut_vals)))]
                q_cut_vals = q_cut_vals[np.intersect1d(np.where(q_cut_vals != 0)[0], np.where(~np.isnan(q_cut_vals)))]

                if len(q_cut_vals > 0):
                    integrated_err = np.sqrt(np.nansum(q_cut_err ** 2)) / len(q_cut_vals)
                    integrated_intensity = np.average(q_cut_vals, weights=1.0 / q_cut_err)
                    integrated_errs.append(integrated_err)
                    integrated_intensities.append(integrated_intensity)
                else:
                    integrated_err = 0
                    integrated_intensity = 0
                    integrated_errs.append(integrated_err)
                    integrated_intensities.append(integrated_intensity)

            q_vals = q
            binned_intensities = integrated_intensities
            binned_errors = integrated_errs
            bin_x = q_vals
            bin_y = binned_intensities
            bin_y_err = binned_errors
            # Now bin the cut as specified by the extents array
            extent_res = np.abs(qbins[1] - qbins[0])
            bins = np.arange(qbins[0], qbins[1] + qbins[2] / 2.0, qbins[2])
            bin_x, bin_y, bin_y_err = bin1D(q, bin_y, bin_y_err, bins, statistic='mean')
        elif len(ebins) == 3:
            # First restrict range across Q
            q_slice = intensities[
                np.intersect1d(np.where(q >= qbins[0]), np.where(q <= qbins[1]))]
            slice_errs = errors[
                np.intersect1d(np.where(q >= qbins[0]), np.where(q <= qbins[1]))]
            # Integrate over E for all values of Q
            integrated_intensities = []
            integrated_errs = []
            for i in range(len(q_slice[0])):
                e_cut_vals = q_slice[:, i]
                e_cut_err = slice_errs[:, i]
                e_cut_err = e_cut_err[np.intersect1d(np.where(e_cut_vals != 0)[0], np.where(~np.isnan(e_cut_vals)))]
                e_cut_vals = e_cut_vals[np.intersect1d(np.where(e_cut_vals != 0)[0], np.where(~np.isnan(e_cut_vals)))]

                if len(e_cut_vals) > 0:
                    integrated_err = np.sqrt(np.nansum(e_cut_err ** 2)) / len(e_cut_vals)
                    integrated_intensity = np.average(e_cut_vals, weights=1.0 / e_cut_err)
                    integrated_errs.append(integrated_err)
                    integrated_intensities.append(integrated_intensity)
                else:
                    integrated_errs.append(0)
                    integrated_intensities.append(0)
            bin_x = e
            bin_y = integrated_intensities
            bin_y_err = integrated_errs

            bins = np.arange(ebins[0], ebins[1] + ebins[2] / 2.0, ebins[2])
            bin_x, bin_y, bin_y_err = bin1D(e, bin_y, bin_y_err, bins, statistic='mean')
        else:
            print('Invalid axis option (Use \'|Q|\' or \'DeltaE\')')
            return False

        return np.array(bin_x), np.array(bin_y), np.array(bin_y_err)

    def bosesubtract(self,highTMDwrapper):
        """
        Using a bose-einstein method, subtracts a high temperature measurement from this one.
        :param highTMDwrapper: second high temperature MDwrapper object to subtract
        :return:
        """
        lowmdhisto = CloneWorkspace(self.mdhisto,OutputWorkspace='tempMD_lowT')

        submdhisto = tempsubtractMD(self,highTMDwrapper)
        self.mdhisto=submdhisto
        return None