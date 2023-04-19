class Mat:
    """
    Class for calculation of scattering properties of materials from CIF files.
    Requires asdfa cif file to initialize, as well as the NIST table of scattering data

    :param cif_file: Requisite .cif file. :param nist_data: File where nist database is imported, should be left
    alone.
    :type cif_file: str
    :param b_dict: Scattering dictionary which may be used to manually specify scattering
    properties. Will overwrite the dictionary imported from the NIST values. Must be in the same form as the NIST
    table, where each entry is a list of: [ Isotope, Natural Concentration, Coh b (fm), Inc b (fm), Coh xs (barn),
    Inc xs (barn), Scatt xs (barn), Abs xs (barn]
    :type b_dict: dict
    :param suppress_print: Boolean specifying if the output of a cif import should be printed to console or not.
    :type suppress_print: bool

    """

    def __init__(self, cif_file, nist_data='nist_scattering_table.txt', b_dict=False, suppress_print=False):
        # Initializes the class
        if b_dict == False:
            scatt_dict = import_NIST_table(nist_data)
            self.b_arr = False
        else:
            scatt_dict = b_dict
            self.b_arr = True
            self.scatt_dict = b_dict
        # Make a dictionary of the unique atomic positions from the cif file, get their scattering lengths
        cif_f = open(cif_file, 'r')
        f_lines = cif_f.readlines()
        cif_f.close()
        cif_obj = get_cif_dict(cif_file)

        # The cif obj contains all relevant information.

        # Collect the a, b, c, alpha, beta, gamma values.
        a = float(cif_obj['_cell_length_a'].split('(')[0])
        b = float(cif_obj['_cell_length_b'].split('(')[0])
        c = float(cif_obj['_cell_length_c'].split('(')[0])
        alpha = float(cif_obj['_cell_angle_alpha'].split('(')[0])
        beta = float(cif_obj['_cell_angle_beta'].split('(')[0])
        gamma = float(cif_obj['_cell_angle_gamma'].split('(')[0])
        cell_vol = float(cif_obj['_cell_volume'].split('(')[0])
        # Generate reciprocal lattice
        alpha_r = alpha * np.pi / 180.0
        beta_r = beta * np.pi / 180.0
        gamma_r = gamma * np.pi / 180.0
        avec = np.array([a, 0.0, 0.0])
        bvec = np.array([b * np.cos(gamma_r), b * np.sin(gamma_r), 0])
        cvec = np.array(
            [c * np.cos(beta_r), c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / (np.sin(gamma_r)), \
             c * np.sqrt(1.0 - np.cos(beta_r) ** 2 - (
                         (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)) ** 2)])
        V_recip = np.dot(avec, np.cross(bvec, cvec))
        astar = np.cross(bvec, cvec) / V_recip
        bstar = np.cross(cvec, avec) / V_recip
        cstar = np.cross(avec, bvec) / V_recip

        # The parameter that defines the space group is often inconsistent. Find something that contains the string
        # '_space_group' but not 'xyz'
        space_group = 'Undefined'
        for i in range(len(cif_obj)):
            key_str = cif_obj.keys()[i]
            if (('_space_group' in key_str) or ('_space_group_name_h-m' in key_str)) and ('xyz' not in key_str) and (
                    'number' not in key_str) and ('symop_id' not in key_str):
                # found the key to the space group in the dictionary.
                space_key = key_str
                space_group = cif_obj[key_str]
                continue
        self.avec = avec
        self.bvec = bvec
        self.cvec = cvec
        self.u = astar
        self.v = bstar
        self.w = cstar
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cell_vol = cell_vol
        self.astar = np.linalg.norm(astar)
        self.bstar = np.linalg.norm(bstar)
        self.cstar = np.linalg.norm(cstar)
        self.cell_vol_recip = V_recip
        self.space_group = space_group
        self.fname = cif_file
        self.nist_file = nist_data
        self.scatt_dict = scatt_dict
        self.cif_dict = cif_obj
        # extracts some additional info in the cif file about the general unit cell
        f_lines = self.gen_flines()

        chem_sum = cif_obj['_chemical_formula_sum']
        try:
            formula_weight = float(cif_obj['_chemical_formula_weight'])
        except:
            # Need to calculate from chemsum, not implemented. yet..
            print(
                "WARNING: Chemical weight not in cif file. Placeholder value used but should be updated manually using: \n Material.formula_weight=(val)")
            formula_weight = 100.0

        formula_units = int(cif_obj['_cell_formula_units_Z'])
        self.chem_sum = chem_sum
        self.formula_weight = formula_weight
        self.formula_units = formula_units
        if suppress_print == False:
            print('\n######################\n')
            print('a = ' + str(self.a) + ' Ang')
            print('b = ' + str(self.b) + ' Ang')
            print('c = ' + str(self.c) + ' Ang')
            print('alpha = ' + str(self.alpha) + ' Ang')
            print('beta = ' + str(self.beta) + ' Ang')
            print('gamma = ' + str(self.gamma) + ' Ang')
            print(self.space_group)
            print('Space group: ' + self.space_group)
            print('Unit Cell Volume =' + str(self.cell_vol))
            print('Formula weight = ' + str(self.formula_weight))
            print('Formula units per unit cell = ' + str(self.formula_units))
            print(cif_file + ' imported successfully.' + '\n')
            print('###################### \n')

    def gen_flines(self):
        # simply retuns the flines string array
        cif_f = open(self.fname, 'r')
        f_lines = cif_f.readlines()
        cif_f.close()
        return f_lines

    def generate_symmetry_operations(self):
        # Collect the symmetry equivalent sites. File format is 'site_id, symmetry equiv xyz'
        # Check for sure where the position is
        # returns the symmetry operations as array of strings, i.e. ['+x','+y','-z']

        symm_arr = self.cif_dict['_symmetry_equiv_pos_as_xyz']
        for i in range(len(symm_arr)):
            if type(symm_arr[i]) == str:
                symm_arr[i] = symm_arr[i].split(',')
            else:
                pass
        # Now operations in format of ['+x','+y','+z']
        return symm_arr

    def gen_unique_coords(self):
        # Get the relevant atomic coordinates and displacement aprameters for unique positions
        f_lines = self.gen_flines()

        coords = {}
        # Make a dictionary of the ions and their positions- dctionary of format
        # ['IonLabel': '['ion', fract_x, fract_y, fract_z, occupancy, Uiso, Uiso_val, Multiplicity]]

        atom_site_type_symbol_arr = self.cif_dict['_atom_site_type_symbol']
        atom_site_label_arr = self.cif_dict['_atom_site_label']
        atom_site_fract_x_arr = self.cif_dict['_atom_site_fract_x']
        atom_site_fract_y_arr = self.cif_dict['_atom_site_fract_y']
        atom_site_fract_z_arr = self.cif_dict['_atom_site_fract_z']
        atom_site_occupancy_arr = self.cif_dict['_atom_site_occupancy']
        # The following may or may not be in the cif file
        try:
            atom_site_thermal_displace_type_arr = self.cif_dict['_atom_site_thermal_displace_type']
        except:
            atom_site_thermal_displace_type_arr = np.zeros(len(atom_site_type_symbol_arr))
        try:
            atom_site_U_iso_or_equiv_arr = self.cif_dict['_atom_site_U_iso_or_equiv']
        except:
            atom_site_U_iso_or_equiv_arr = np.zeros(len(atom_site_type_symbol_arr))
        try:
            atom_site_symmetry_multiplicity_arr = self.cif_dict['_atom_site_multiplicity']
        except:
            atom_site_symmetry_multiplicity_arr = np.zeros(len(atom_site_type_symbol_arr))
        for i in range(len(atom_site_label_arr)):
            ion = atom_site_type_symbol_arr[i]
            label = atom_site_label_arr[i]
            fract_x = atom_site_fract_x_arr[i]
            fract_y = atom_site_fract_y_arr[i]
            fract_z = atom_site_fract_z_arr[i]
            occupancy = atom_site_occupancy_arr[i]
            thermal_displace_type = atom_site_thermal_displace_type_arr[i]
            Uiso = atom_site_U_iso_or_equiv_arr[i]
            sym_mult = atom_site_symmetry_multiplicity_arr[i]
            coords[label] = [ion, fract_x, fract_y, fract_z, occupancy, thermal_displace_type, Uiso, sym_mult]
        # Note these are all still strings

        # The cif file tells the number of atom types in the cell- we can check the results of our symmetry operations with this.
        expected = {}
        try:
            ion_list = self.cif_dict['_atom_type_symbol']
            ion_number = self.cif_dict['_atom_type_number_in_cell']
        except:
            ion_list = [0]
            ion_number = [0]
        for i in range(len(ion_list)):
            expected[ion_list[i]] = float(ion_number[i])
        # store this information for later
        self.expected_sites = expected
        return coords

    def gen_unit_cell_positions(self):
        # Now we must generate all of the positions. We'll generate a list of format:
        #    [ion, x, y, z]
        # which makes structure factor calclation easier.
        coords = self.gen_unique_coords()
        scatt_dict = self.scatt_dict
        symm_ops = self.generate_symmetry_operations()

        positions = []
        structure_array = []
        # for every ion, iterate over the symmetry operations. If the position already exists, it is not unique and not added to the posisions array
        for position in coords:
            ion_coords = coords[position]
            ion = ion_coords[0]
            # For some reason O has a - next to it..
            ion = ion.replace('-', '')
            ion = ion.replace('+', '')
            ion = "".join([x for x in ion if x.isalpha()])
            x = float(ion_coords[1].replace(' ', '').split('(')[0])
            y = float(ion_coords[2].replace(' ', '').split('(')[0])
            z = float(ion_coords[3].replace(' ', '').split('(')[0])
            try:
                if self.b_arr == False:
                    b_el = float(scatt_dict[ion][1])
                else:
                    b_el = float(scatt_dict[ion])
            except KeyError:
                print(
                    'Ion ' + ion + ' not found in NIST Tables or included b_arr. Include argument b_arr with elastic scattering lengths in fm when declaring Material object.')
                break
                return 0
            occupancy = float(ion_coords[4])

            for j in range(len(symm_ops)):
                symmetry = symm_ops[j]
                # replace the x with our new x, etc for y and z
                x_sym = symmetry[0]
                x_eval_str = x_sym.replace('x', str(x))
                x_eval_str = x_eval_str.replace('y', str(y))
                x_eval_str = x_eval_str.replace('z', str(z))
                x_eval_str = x_eval_str.replace('/', '*1.0/')
                y_sym = symmetry[1]
                y_eval_str = y_sym.replace('x', str(x))
                y_eval_str = y_eval_str.replace('y', str(y))
                y_eval_str = y_eval_str.replace('z', str(z))
                y_eval_str = y_eval_str.replace('/', '*1.0/')

                z_sym = symmetry[2]
                z_eval_str = z_sym.replace('x', str(x))
                z_eval_str = z_eval_str.replace('y', str(y))
                z_eval_str = z_eval_str.replace('z', str(z))
                z_eval_str = z_eval_str.replace('/', '*1.0/')
                x_pos = eval(x_eval_str)
                y_pos = eval(y_eval_str)
                z_pos = eval(z_eval_str)
                # assume that atoms can be no closer than 0.1 Ang
                z_pos = round(z_pos, 2)
                x_pos = round(x_pos, 2)
                y_pos = round(y_pos, 2)
                if x_pos == 0.0:
                    x_pos = 0.0
                if y_pos == 0.0:
                    y_pos = 0.0
                if z_pos == 0.0:
                    z_pos = 0.0
                if x_pos < 0.0:
                    x_pos += 1.
                if x_pos >= 1.0:
                    x_pos -= 1.
                if y_pos < 0.0:
                    y_pos += 1.
                if y_pos >= 1.00:
                    y_pos -= 1.
                if z_pos < -0.0:
                    z_pos += 1.
                if z_pos >= 1.00:
                    z_pos -= 1.0

                occ = occupancy
                pos = [round(x_pos, 2), round(y_pos, 2), round(z_pos, 2), occ]
                if pos not in positions:
                    positions.append(pos)
                    structure_array.append([ion, b_el, x_pos, y_pos, z_pos, occ])

        # Now we have all of the positions!
        self.unit_cell_xyz = structure_array
        return structure_array

    def plot_unit_cell(self, cmap='jet'):
        # NOTE: only supports up to 10 different atoms.
        structure = self.gen_unit_cell_positions()
        unique_ions = np.unique(np.array(structure)[:, 0])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(unique_ions))
        figure = plt.figure(1, figsize=(8, 8))
        MAX = np.max([self.a, self.b, self.c])
        ax = figure.add_subplot(111, projection='3d')
        used_ions = []
        for i in range(len(structure)):
            x = structure[i][2] * self.a
            y = structure[i][3] * self.b
            z = structure[i][4] * self.c
            b_val = structure[i][1]
            occupancy = structure[i][5]
            ion = structure[i][0]
            ion_i = np.where(unique_ions == ion)[0][0]

            color = np.array(matplotlib.cm.jet(norm(ion_i + 0.5)))

            if ion not in used_ions:
                ax.scatter(x, y, z, c=color, label=ion, s=5.0 * (np.abs(b_val)))
                used_ions.append(ion)
            else:
                ax.scatter(x, y, z, c=color, alpha=occupancy, s=5.0 * (np.abs(b_val)))
        # Weirdly need to plot white points to fix the aspect ratio of the box
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.max([self.a, self.b, self.c])
        mid_x = (self.a) * 0.5
        mid_y = (self.b) * 0.5
        mid_z = (self.c) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.legend()
        plt.show()
        return figure, ax

    def gen_reflection_list(self, max_tau=20, maxQmag=1e10, b_dict=False):
        # Calculates the structure factor for all reflections in the unit cell.
        # returns an array of arrays of format [H K L Freal Fi |F|^2 ]

        # NOTE add in occupancy later
        # Need to convert from fractional to real coordinates
        # Returns in units of per unit cell
        structure = self.gen_unit_cell_positions()

        F_HKL = 0.0
        # Generate array of Q-vectors
        taulim = np.arange(-max_tau + 1, max_tau)
        xx, yy, zz = np.meshgrid(taulim, taulim, taulim)
        x = xx.flatten()
        y = yy.flatten()
        z = zz.flatten()
        # array of reciprocal lattice vectors; 4th column will be structure factor^2
        tau = np.array([x, y, z, np.zeros(len(x))]).transpose()

        ion_list = np.array(structure)[:, 0]
        occupancy_arr = np.array(structure)[:, 5].astype(float)
        b_array = occupancy_arr * np.array(structure)[:, 1].astype(float)

        # Imported from the NIST site so this is in femtometers, for result to be barn divide by 10
        b_array = b_array

        unit_cell_pos = np.array(structure)[:, 2:5].astype(float)

        a_vec = self.avec
        b_vec = self.bvec
        c_vec = self.cvec
        u_vec = self.u
        v_vec = self.v
        w_vec = self.w

        i = 0
        bad_ind = []
        for i in range(len(tau)):
            q_vect = tau[i][0:3]
            qmag = self.Qmag_HKL(q_vect[0], q_vect[1], q_vect[2])

            if qmag > maxQmag:
                bad_ind.append(i)
                tau[i, 3] = 0.0
            else:
                SF = 0
                # Sum over all ions in the cell
                for j in range(len(unit_cell_pos)):
                    pos = unit_cell_pos[j]
                    SF = SF + occupancy_arr[j] * b_array[j] * np.exp(2.0j * np.pi * np.inner(q_vect, pos))
                tau[i, 3] = np.linalg.norm(SF) ** 2 / 100.0  # Fm^2 to barn
        # Eliminate tiny values
        tau[:, 3][tau[:, 3] < 1e-8] = 0.0
        low_reflect_i = np.where(tau[:, 3] == 0.0)[0]
        zero_ind = np.where(tau[:, 3] == 0.0)[0]
        # Divide but the number of formula units per unit cell to put it in units of barn / f.u.
        tau[:, 3] = tau[:, 3]  # /self.formula_units
        self.HKL_list = tau
        return tau

    def calc_F_HKL(self, H, K, L):
        # Directly calculates a SF^2 for an arbitrary HKL index
        structure = self.gen_unit_cell_positions()

        F_HKL = 0.0
        # Generate array of Q-vectors
        tau = np.array([H, K, L, 0])

        ion_list = np.array(structure)[:, 0]
        occupancy_arr = np.array(structure)[:, 5].astype(float)
        b_array = occupancy_arr * np.array(structure)[:, 1].astype(float)

        # Imported from the NIST site so this is in femtometers, for result to be barn divide by 10
        b_array = b_array * 0.1

        unit_cell_pos = np.array(structure)[:, 2:5].astype(float)

        a_vec = self.avec
        b_vec = self.bvec
        c_vec = self.cvec
        u_vec = self.u
        v_vec = self.v
        w_vec = self.w

        i = 0
        bad_ind = []
        SF = 0
        for j in range(len(unit_cell_pos)):
            q_vect = np.array([H, K, L])
            pos = unit_cell_pos[j]
            SF = SF + occupancy_arr[j] * b_array[j] * np.exp(2.0j * np.pi * np.inner(q_vect, pos))
        tau[3] = np.linalg.norm(SF) ** 2
        # Divide but the number of formula units per unit cell to put it in units of barn / f.u.
        return tau

    def fetch_F_HKL(self, H, K, L):
        # Returns the SF^2 of a particular reflection
        try:
            HKL = self.HKL_list
        except AttributeError:
            HKL = self.gen_reflection_list()
        index = np.argmin(np.abs(H - HKL[:, 0]) + np.abs(K - HKL[:, 1]) + np.abs(L - HKL[:, 2]))
        # print('|F|^2 for ['+str(H)+str(K)+str(L)+'] ='+str(HKL[index][3])+' (fm^2/sr)')
        return HKL[index]

    def Qmag_HKL(self, H, K, L):
        # Returns the magnitude of the q-vector of the assosciated HKL indec in Ang^-1
        qvec = 2.0 * np.pi * np.array(H * self.u + K * self.v + L * self.w)
        qmag = np.linalg.norm(qvec)
        return qmag

    def twotheta_hkl(self, H, K, L, E, mode='deg'):
        # Simply feeds into equivalent Q_HKL function then converts to twotheta
        qmag = self.Qmag_HKL(H, K, L)
        lamb = 9.045 / np.sqrt(E)
        twotheta = np.arcsin(qmag * lamb / (4.0 * np.pi)) * 2.0
        if mode == 'deg':
            return twotheta * 180.0 / np.pi
        else:
            return twotheta

    def theory_bragg_I(self, obsQ, intQ, intE, H, K, L, sample_mass):
        # Returns a scale factor to scale the data into  'barn/(eV mol sr fu)'
        # Makes a very rough assumption that Q^2 is constant through the integral (clearly it's not)
        # If you want to avoid this you need to evaluate int(Q^2), and input 1.0 as obsQ

        '''
        Input params:
            obsQ - center of bragg peak
            intQ - integral in Q of bragg peak
            intE - integral in E of bragg peak (In meV!!)
            HKL, indices of bragg peak
            sample_mass- mass of sample in grams

        Returns:
            A scaling factor to normalize your dataset to the bragg peak.In units of fm^2
            Multiply by multiplicity afterwards if needed
        '''
        observed_Qsqr = obsQ ** 2 * (intQ)
        obs_E_int = intE  # To become eV rather than meV
        I_obs = observed_Qsqr * obs_E_int
        f_HKL = self.fetch_F_HKL(H, K, L)[-1]
        # Convert to barn^2 / Sr
        f_HKL = f_HKL  # *0.01
        # We want it per fu, so scale the fHKL to reflect this
        # f_HKL/=formula_scale
        density = self.formula_weight
        N = sample_mass / density
        numerator = (4.0 * np.pi) * I_obs * N
        denom = f_HKL * (((2 * np.pi) ** 3) / self.cell_vol)
        scaling_factor = denom / numerator

        return scaling_factor

    def calc_sample_absorption(self, Ei, deltaE, d_eff, abs_dict=False, suppress_print=False):
        # Given a value or array of Ei and deltaE, returns the absorption per formula unit for the material.
        # Also requires an effective distance

        # Can override the tabulated absorption if a dictionary in the format of abs_dict=['ion_str':absorption cross section] is give
        if abs_dict == False:
            scatt_dict = import_NIST_table(self.nist_file)

        lambda_i = np.sqrt(81.81 / Ei)
        lambda_f = np.sqrt(81.81 / (Ei - deltaE))
        lambda0 = 3956.0 / 2200.0
        rI = lambda_i / lambda0
        rF = lambda_f / lambda0
        cell_V = self.cell_vol
        formula = self.chem_sum
        num_units = self.formula_units
        formula_list = formula.split()
        atoms = []
        for string in formula_list:
            num = ''.join(x for x in string if x.isdigit())
            ion = ''.join(x for x in string if x.isalpha())
            if not num:
                num = 1
            atoms.append([ion, int(num)])
        sigma_abs = 0.0
        for i in range(len(atoms)):
            if abs_dict == False:
                abs_xc_str = self.scatt_dict[atoms[i][0]][-1]

                abs_xc = float(abs_xc_str.split('(')[0])
                sigma_abs = sigma_abs + (atoms[i][1] * abs_xc)
            else:
                abs_xc = abs_dict[atoms[i][0]]
                sigma_abs = sigma_abs + (atoms[i][1] * abs_xc)
        self.rho_abs = sigma_abs
        sigma_abs = sigma_abs * num_units
        if suppress_print == False:
            print('Mean elastic path length for Ei=' + str(round(Ei, 2)) + 'meV = ' + str(
                round(1.0 / (sigma_abs * rI / cell_V), 2)) + ' cm')
        transmission_vs_energy_i = np.exp(-d_eff * sigma_abs * rI / cell_V)
        transmission_vs_energy_f = np.exp(-d_eff * sigma_abs * rF / cell_V)
        geo_mean_tranmission = np.sqrt(transmission_vs_energy_i * transmission_vs_energy_f)

        return geo_mean_tranmission