def import_NIST_table(nist_file='nist_scattering_table.txt'):
    """
    Function to import scattering data for all ions from NIST database.
    :param nist_file: filename downloaded from NIST site.

    :return scatter_dict: Dictionary containing all elements of the scattering file for all available ions and isotopes.
    :rtype: dict
    """
    # Columns within file represent the following:
    #     Isotope 	conc 	Coh b 	Inc b 	Coh xs 	Inc xs 	Scatt xs 	Abs xs
    f = open(nist_file, 'r')
    f_lines = f.readlines()
    f.close()
    scatter_dict = {}
    for i in range(len(f_lines))[1:]:
        # Skipping the first line, append all the results to our dictionary
        line = f_lines[i].strip('\r\n').split('\t')
        line_strip = [element.strip(' ') for element in line]
        element = line_strip[0]
        data = line_strip[1:]
        scatter_dict[element] = data
    return scatter_dict
