"""
Contains low-level routines for crystal structure energy calculations.
"""

import itertools
import numpy

class Cell:
    """
    Creates a unit cell with the specified parameters.

    Parameters
    ----------
    vectors : ndarray
        Cell vectors as a matrix of row vectors (columns set dimensions)
    atom_lists : list of ndarray
        Atom coordinates as matrices of row vectors (one matrix per type)

    Raises
    ------
    ValueError
        Inconsistent dimensions or invalid values of input data
    """
    def __init__(self, vectors, atom_lists):
        # Process the vector list and retrieve the number of dimensions
        self.__vectors = numpy.array(vectors, dtype=float, order="C")
        if self.__vectors.ndim != 2 or self.__vectors.shape[0] != self.__vectors.shape[1]:
            raise ValueError("vectors must be provided as a square matrix")
        self.__dimensions = self.__vectors.shape[1]
        if numpy.linalg.matrix_rank(self.__vectors) != self.__dimensions:
            raise ValueError("vectors must form a cell with non-zero volume")

        # Process the atom lists containing coordinates of atoms of different types
        self.__atom_lists = []
        for atom_list in atom_lists:
            atom_list = numpy.array(atom_list, dtype=float, order="C")
            if atom_list.ndim != 2 or atom_list.shape[1] != self.__dimensions:
                raise ValueError("atom coordinates must be consistent with cell dimensions")
            self.__atom_lists.append(atom_list)
        self.__atom_types = len(self.__atom_lists)

    def dimensions(self):
        """
        Retrieves the number of spatial dimensions of the cell.

        Returns
        -------
        int
            Number of dimensions
        """

        return self.__dimensions
    
    def vector(self, index):
        """
        Retrieves a cell vector.

        Parameters
        ----------
        index : int
            Index of cell vector to retrieve

        Returns
        -------
        ndarray
            Row vector requested
        """

        return self.__vectors[index].copy()

    def atom_types(self):
        """
        Retrieves the number of atom types.

        Returns
        -------
        int
            Number of types
        """

        return self.__atom_types

    def atom_count(self, index):
        """
        Retrieves the number of atoms of a given type.

        Parameters
        ----------
        index : int
            Index of atom type to examine

        Returns
        -------
        int
            Number of atoms requested
        """

        return self.__atom_lists[index].shape[0]

    def atom(self, list_index, atom_index):
        """
        Retrieves the coordinates of a given atom.

        Parameters
        ----------
        list_index : int
            Index of atom type to examine
        atom_index : int
            Index of atom to examine

        Returns
        -------
        ndarray
            Row vector of coordinates requested
        """

        return self.__atom_lists[list_index][atom_index].copy()
    
    def histogram(self, source_list_index, target_list_index):
        """
        Retrieves a pregenerated histogram.

        Parameters
        ----------
        source_list_index : int
            Source atom type index
        target_list_index : int
            Target atom type index

        Returns
        -------
        dict of float -> int
            Histogram as (distance -> atom_count)
        """

        return dict(self.__histograms[source_list_index][target_list_index])

    def contact(self, source_list_index, target_list_index):
        """
        Retrieves pregenerated atom contact information.

        Parameters
        ----------
        source_list_index : int
            Source atom type index
        target_list_index : int
            Target atom type index

        Returns
        -------
        float
            Minimum distance between atoms
        """

        return self.__contacts[source_list_index][target_list_index]

    def write_data(self, text_file):
        """
        Writes vector and atom information to a text file.

        Parameters
        ----------
        text_file : TextIOBase
            Output file
        """

        # Write the heading line
        text_file.write("{} {}\n".format(self.__dimensions, " ".join(str(atom_list.shape[0]) \
            for atom_list in self.__atom_lists)))

        # Write vector data
        for vector_index in range(self.__dimensions):
            text_file.write("{}\n".format(" ".join(str(float(self.__vectors[vector_index, coordinate_index])) \
                for coordinate_index in range(self.__dimensions))))

        # Write atom data
        for atom_list in self.__atom_lists:
            for atom_index in range(atom_list.shape[0]):
                text_file.write("{}\n".format(" ".join(str(float(atom_list[atom_index, coordinate_index])) \
                    for coordinate_index in range(self.__dimensions))))

    def write_lammps(self, text_file):
        """
        Writes vector and atom information to a LAMMPS file.

        Parameters
        ----------
        text_file : TextIOBase
            Output file

        Raises
        ------
        NotImplementedError
            The cell does not have exactly 3 dimensions
        """

        # Check the number of dimensions
        if self.__dimensions != 3:
            raise NotImplementedError("LAMMPS export supported in 3D only")
        
        # Write header information
        text_file.write("LAMMPS\n\n{} atoms\n{} atom types\n\n".format(
            sum(atom_list.shape[0] for atom_list in self.__atom_lists), len(self.__atom_lists)))

        # Retrieve cell vectors
        A, B, C = self.__vectors

        # Calculate new vector components
        ax = numpy.linalg.norm(A)
        bx = numpy.dot(B, A / ax)
        by = numpy.sqrt(numpy.dot(B, B) - (bx * bx))
        cx = numpy.dot(C, A / ax)
        cy = (numpy.dot(B, C) - (bx * cx)) / by
        cz = numpy.sqrt(numpy.dot(C, C) - (cx * cx) - (cy * cy))

        # Write box data
        text_file.write("0 {} xlo xhi\n".format(ax))
        text_file.write("0 {} ylo yhi\n".format(by))
        text_file.write("0 {} zlo zhi\n".format(cz))
        text_file.write("{} {} {} xy xz yz\n\nAtoms\n\n".format(bx, cx, cy))

        # Create transformation matrix
        a = numpy.array([ax, 0.0, 0.0])
        b = numpy.array([bx, by, 0.0])
        c = numpy.array([cx, cy, cz])
        volume = numpy.dot(A, numpy.cross(B, C))
        M = numpy.array([a, b, c]).T @ numpy.array([numpy.cross(B, C), numpy.cross(C, A), numpy.cross(A, B)]) \
            / numpy.dot(A, numpy.cross(B, C))

        # Write actual atom positions
        atom_counter = 0
        for list_index, atom_list in enumerate(self.__atom_lists):
            for atom_index in range(atom_list.shape[0]):
                text_file.write("{} {} {} {} {}\n".format(atom_counter + 1, list_index + 1, *(M @ atom_list[atom_index])))
                atom_counter += 1

    @staticmethod
    def read_data(text_file):
        """
        Reads vector and atom information from a text file.

        Parameters
        ----------
        text_file : TextIOBase
            Input file

        Returns
        -------
        Cell
            Data read

        Raises
        ------
        ValueError
            Invalid data were encountered
        """

        lines = (line for line in (line.strip() for line in text_file) if line)

        # Read the heading line
        lengths = [int(value.strip()) for value in next(lines).split()]
        dimensions, list_lengths = lengths[0], lengths[1:]
        if dimensions < 0:
            raise ValueError("number of dimensions must not be negative")
        for list_length in list_lengths:
            if list_length < 0:
                raise ValueError("number of atoms must not be negative")

        # Read vector data
        vectors = [None] * dimensions
        for vector_index in range(dimensions):
            vector = [float(value.strip()) for value in next(lines).split()]
            if len(vector) != dimensions:
                raise ValueError("vector coordinates must be consistent with cell dimensions")
            vectors[vector_index] = vector

        # Read atom data
        atom_lists = [None] * len(list_lengths)
        for list_index in range(len(list_lengths)):
            atom_list = [None] * list_lengths[list_index]
            for atom_index in range(list_lengths[list_index]):
                coordinates = [float(value.strip()) for value in next(lines).split()]
                if len(coordinates) != dimensions:
                    raise ValueError("atom coordinates must be consistent with cell dimensions")
                atom_list[atom_index] = coordinates
            atom_lists[list_index] = atom_list

        # Ensure that the end of the file has been reached
        try:
            next(lines)
            raise ValueError("unexpected additional data encountered")
        except StopIteration:
            pass
 
        return Cell(vectors, atom_lists)

    def write_histogram(self, text_file):
        """
        Writes histogram information to a text file.

        Parameters
        ----------
        text_file : TextIOBase
            Output file
        """

        # Write the heading line
        text_file.write("{}\n".format(" ".join(str(len(self.__histograms[source_list_index][target_list_index])) \
            for source_list_index in range(self.__atom_types) for target_list_index in range(self.__atom_types))))

        # Write histogram data
        for source_list_index in range(self.__atom_types):
            for target_list_index in range(self.__atom_types):
                histogram = self.__histograms[source_list_index][target_list_index]
                for distance in sorted(histogram):
                    text_file.write("{} {}\n".format(distance, histogram[distance]))

    def read_histogram(self, text_file):
        """
        Reads histogram information from a text file.

        Parameters
        ----------
        text_file : TextIOBase
            Input file

        Raises
        ------
        ValueError
            Invalid data were encountered
        """

        lines = (line for line in (line.strip() for line in text_file) if line)

        # Read the heading line
        lengths = [int(value.strip()) for value in next(lines).split()]
        if len(lengths) != self.__atom_types ** 2:
            raise ValueError("number of histograms must be consistent with number of atom types")
        for length in lengths:
            if length < 0:
                raise ValueError("number of histogram entries must not be negative")

        histograms = [[None] * self.__atom_types for list_index in range(self.__atom_types)]

        # Read the data
        length_index = 0
        for source_list_index in range(self.__atom_types):
            for target_list_index in range(self.__atom_types):
                histogram = {}
                for line_index in range(lengths[length_index]):
                    key, value = (value.strip() for value in next(lines).split())
                    key, value = float(key), int(value)
                    if key in histogram:
                        raise ValueError("duplicate histogram entry encountered")
                    histogram[key] = value
                length_index += 1

                histograms[source_list_index][target_list_index] = histogram

        # Ensure that the end of the file has been reached
        try:
            next(lines)
            raise ValueError("unexpected additional data encountered")
        except StopIteration:
            pass

        self.__histograms = histograms
        self.__histogram_update()

    def measure(self, cutoff):
        """
        Performs histogram measurements on the system out to a given cutoff distance.

        Parameters
        ----------
        cutoff : float
            Cutoff distance

        Raises
        ------
        ValueError
            No atoms were identified within the specified cutoff
        """

        histograms = [[None] * self.__atom_types for list_index in range(self.__atom_types)]

        for source_list_index in range(self.__atom_types):
            for target_list_index in range(self.__atom_types):
                histogram = self.__histogram(source_list_index, target_list_index, cutoff)
                if not histogram:
                    raise ValueError("no atoms of a given type found within cutoff")
 
                histograms[source_list_index][target_list_index] = histogram

        self.__histograms = histograms
        self.__histogram_update()

    def __histogram_update(self):
        """
        Updates auxiliary data associated with histograms.
        """

        contacts = [[None] * self.__atom_types for list_index in range(self.__atom_types)]
        key_arrays = [[None] * self.__atom_types for list_index in range(self.__atom_types)]
        value_arrays = [[None] * self.__atom_types for list_index in range(self.__atom_types)]

        for source_list_index in range(self.__atom_types):
            for target_list_index in range(self.__atom_types):
                histogram = self.__histograms[source_list_index][target_list_index]
                contacts[source_list_index][target_list_index] = min(histogram)
                key_arrays[source_list_index][target_list_index] = numpy.array(list(histogram.keys()))
                value_arrays[source_list_index][target_list_index] = numpy.array(list(histogram.values()))

        self.__contacts = contacts
        self.__key_arrays = key_arrays
        self.__value_arrays = value_arrays

    def __histogram(self, source_list_index, target_list_index, cutoff):
        """
        Performs a histogram measurement on two atom lists out to a given cutoff distance.

        Parameters
        ----------
        source_list_index : int
            Source atom type index
        target_list_index : int
            Target atom type index
        cutoff : float
            Cutoff distance

        Returns
        -------
        dict of float -> int
            Calculated histogram as (distance -> atom_count)
        """

        # Initialize
        source_list = self.__atom_lists[source_list_index]
        target_list = self.__atom_lists[target_list_index]
        histogram = {}
        cell_index = 0
        minimum = 0.0

        # Continue scanning outwards as long as the minimum distance is too low
        while minimum < cutoff:
            # Reset the minimum distance to find it for just this "shell" of cells
            minimum = 0.0

            for cell_coordinates in itertools.product(*(range(-cell_index, cell_index + 1) \
                for dimension_index in range(self.__dimensions))):

                # Ignore cells not in the desired "shell"
                if numpy.max(numpy.abs(cell_coordinates)) != cell_index:
                    continue

                # Perform the calculations for the current cell
                shift_vector = numpy.sum(self.__vectors * numpy.repeat(cell_coordinates, self.__dimensions) \
                    .reshape(self.__dimensions, self.__dimensions), 0)
                for source_atom in source_list:
                    for target_atom in target_list:
                            distance = numpy.linalg.norm(shift_vector + target_atom - source_atom)
                            if not distance:
                                continue
                            if not minimum or minimum > distance:
                                minimum = distance
                            if distance in histogram:
                                histogram[distance] += 1
                            else:
                                histogram[distance] = 1

            # Move to the next "shell" of cells
            cell_index += 1

        # Filter out atoms past the cutoff
        return { distance: histogram[distance] for distance in histogram if distance <= cutoff }

    def energy(self, potentials, parameter_sets):
        """
        Determines the interaction energy per atom of the unit cell.

        Parameters
        ----------
        potentials : dict of (tuple of int), potential
            Potentials for each pair of atom types
        parameter_sets : list of dict of (tuple of int), tuple
            Sets of parameters fed to the potentials

        Returns
        -------
        list of float
            Energies calculated for each parameter set

        Raises
        ------
        ValueError
            Duplicate specification of atom type pairs was encountered
        """

        # Generate a potential array
        zero_potential = lambda r, *parameter_set: 0.0
        potential_array = [[zero_potential] * self.__atom_types for list_index in range(self.__atom_types)]
        for pair in potentials:
            source_list_index, target_list_index = pair
            if potential_array[source_list_index][target_list_index] is not zero_potential:
                raise ValueError("duplicate pair potential assignment encountered")
            potential_array[source_list_index][target_list_index] = potentials[pair]
            if source_list_index != target_list_index:
                potential_array[target_list_index][source_list_index] = potentials[pair]

        # Determine the potential energies for the given parameter sets
        energies = []
        for parameter_set in parameter_sets:
            # Generate a parameter array
            no_parameters = ()
            parameter_array = [[no_parameters] * self.__atom_types for list_index in range(self.__atom_types)]
            for pair in parameter_set:
                source_list_index, target_list_index = pair
                if parameter_array[source_list_index][target_list_index] is not no_parameters:
                    raise ValueError("duplicate parameter assignment encountered")
                parameter_array[source_list_index][target_list_index] = parameter_set[pair]
                if source_list_index != target_list_index:
                    parameter_array[target_list_index][source_list_index] = parameter_set[pair]
            
            # Calculate the energies
            energy = 0.0
            for source_list_index in range(self.__atom_types):
                for target_list_index in range(self.__atom_types):
                    energy += 0.5 * sum(self.__value_arrays[source_list_index][target_list_index] * \
                        potential_array[source_list_index][target_list_index](self.__key_arrays[source_list_index][target_list_index], \
                        *parameter_array[source_list_index][target_list_index]))
            energies.append(energy / sum(atom_list.shape[0] for atom_list in self.__atom_lists))
        return energies
    
    def rescale(self, vectors):
        """
        Rescales a unit cell by applying new basis vectors.

        Parameters
        ----------
        vectors : ndarray
            An array of basis vectors.

        Raises
        ------
        ValueError
            Invalid basis vectors were received
        """
        # Check the vectors
        vectors = numpy.array(vectors, dtype=float, order="C")
        if vectors.shape != self.__vectors.shape:
            raise ValueError("vectors must be provided as a square matrix")
        if numpy.linalg.matrix_rank(vectors) != self.__dimensions:
            raise ValueError("vectors must form a cell with non-zero volume")
        inverse = numpy.linalg.inv(self.__vectors.T)
        self.__vectors = vectors
        
        # Adjust the coordinates
        self.__atom_lists = [(inverse @ atom_list.T).T @ self.__vectors for atom_list in self.__atom_lists]

        # Invalidate any results if they exist
        try:
            del self.__histograms
            del self.__contacts
            del self.__key_arrays
            del self.__value_arrays
        except AttributeError:
            pass
