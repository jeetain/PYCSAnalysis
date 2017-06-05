"""
Performs calculations to determine energies of a large number of crystal cells with a large number of potential parameters.
"""

import itertools
import numpy
import scipy.optimize

def rescale(cells):
    """
    Rescales cells such that the minimum contact distance is unity.

    Parameters
    ----------
    cells : list of :py:class:`crystal.Cell`
        Cells to rescale in place
    """
    for cell in cells:
        # Determine the maximum length diagonal in the unit cell
        maximum_length = max(numpy.linalg.norm(sum(factors[dimension_index] * cell.vector(dimension_index) \
            for dimension_index in range(cell.dimensions()))) \
            for factors in itertools.product(*([-1.0, 1.0] for dimension_index in range(cell.dimensions()))))

        # Measure the cell out to the cutoff distance
        cell.measure(maximum_length)

        # Determine the minimum interatomic distance and rescale
        minimum_distance = min(cell.contact(source_list_index, target_list_index) \
            for source_list_index in range(cell.atom_types()) for target_list_index in range(cell.atom_types()))
        cell.rescale([cell.vector(dimension_index) / minimum_distance for dimension_index in range(cell.dimensions())])

def minima(potentials, parameter_sets, bounds):
    """
    Determines the positions of minima for a number of potentials and parameter sets.

    Parameters
    ----------
    potentials : dict of ? -> potential
        Potentials identified by arbitrary keys
    parameter_sets : list of dict of ? -> potential
        Sets of parameters fed to the potentials
    bounds : list of dict of ? -> (tuple of float)
        Bounds within which to search for the minima

    Returns
    -------
    list of dict of ? -> float
        Minima calculated for each potential with each parameter set

    Raises
    ------
    RuntimeError
        The minimization algorithm failed
    """
    # Determine the minima for the given parameter sets
    minima = []
    for parameter_set_index, parameter_set in enumerate(parameter_sets):
        # Determine the minima for each potential
        minimum_dict = {}
        for key in potentials:
            # Perform the minimization (xatol=0.0 generates the most accurate solution)
            objective = lambda r: potentials[key](r, *parameter_set[key])
            minimize_result = scipy.optimize.minimize_scalar(objective, method="bounded", \
                bounds=bounds[parameter_set_index][key], options={"xatol": 0.0})
            if not minimize_result.success:
                raise RuntimeError("Convergence failure during potential minimization: {}".format(minimize_result.message))
            minimum_dict[key] = minimize_result.x
        minima.append(minimum_dict)
    return minima

def energies(stoichiometry, cells, potentials, parameter_sets, scale_target=None, search_bounds=None):
    """
    Calculates system energies assuming the formation of a single ordered phase.

    Parameters
    ----------
    stoichiometry : list of float
        System solution stoichiometry
    cells : list of :py:class:`crystal.Cell`
        Structures to analyze
    potentials : dict of (tuple of int) -> potential
        Potentials for each pair of atom types
    parameter_sets : list of dict of (tuple of int) -> tuple
        Sets of parameters fed to the potentials
    scale_target : float
        Optional potential minimum distance target
    search_bounds : list of dict of (tuple of int) -> (tuple of float)
        Bounds in which to search for potential minima

    Returns
    -------
    list of list of float
        Energies per atom for each structure
    """
    # Scale potentials
    if scale_target:
        # Find the minima
        minima_list = minima(potentials, parameter_sets, search_bounds)
        # Modify potentials to automatically retrieve minima from parameter sets
        call_potentials = {}
        for key, potential in potentials.items():
            def call_potential(r, *parameters):
                return potential(r * parameters[0], *parameters[1:])
            call_potentials[key] = call_potential
        # Modify parameter sets to have minima in place
        call_parameter_sets = []
        for index, parameter_set in enumerate(parameter_sets):
            call_parameter_sets.append({key: (minima_list[index][key], *value) for key, value in parameter_set.items()})
    else:
        # Don't modify potentials or parameter sets
        call_potentials = potentials
        call_parameter_sets = parameter_sets

    # Normalize solution stoichiometry
    stoichiometry = [atom_count / sum(stoichiometry) for atom_count in stoichiometry]

    # Process each cell
    energies = []
    for cell in cells:
        # Normalize cell stoichiometry and determine fraction of particles in ordered phase
        cell_stoichiometry = [cell.atom_count(type_index) for type_index in range(cell.atom_types())]
        cell_stoichiometry = [atom_count / sum(cell_stoichiometry) for atom_count in cell_stoichiometry]
        ordered_fraction = min(stoichiometry[type_index] / cell_stoichiometry[type_index] for type_index in range(cell.atom_types()))

        # Calculate the energy
        energies.append([ordered_fraction * energy for energy in cell.energy(call_potentials, call_parameter_sets)])
    return energies
