"""
Domain Module
=============

This module provides functionality to represent the domain for the problem.

Classes:
    Domain: Represents the domain for the problem.
"""

from mpi4py import MPI

from dolfinx import io


class Domain:
    """
    Class representing the domain for the problem.

    This class reads the mesh from a GMSH file and locates physical groups.

    Attributes
    ----------
    mesh : dolfinx.Mesh
        The mesh representing the domain.
    cell_tags : numpy.ndarray
        Array containing cell tags.
    facet_tags : numpy.ndarray
        Array containing facet tags.
    boundary_facets : dict
        Dictionary containing boundary facets grouped by physical group name.
    """

    def __init__(self, mesh_pars, dim):
        """
        Initialize the Domain.

        Parameters
        ----------
        mesh_pars : dict
            Dictionary containing parameters for the mesh.
        dim : int
            Dimension of the problem.
        """
        print("\n████ READING THE MESH")
        # Read the mesh from GMSH
        print("Mesh reading output:")
        msh_file = mesh_pars["msh_file"]
        self.mesh_data = io.gmsh.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)
        self.mesh = self.mesh_data.mesh
        self.cell_tags = self.mesh_data.cell_tags
        self.facet_tags = self.mesh_data.facet_tags
        # Locate the physical groups
        self.__locate_physical_groups(mesh_pars["physical_groups"])

    def __locate_physical_groups(self, facets_tags_values):
        """
        Locate physical groups in the mesh.

        Parameters
        ----------
        facets_tags_values : dict
            Dictionary containing facet tags and their corresponding values.
        """
        # Get the facets indices
        self.boundary_facets = {
            facet_name: self.facet_tags.indices[self.facet_tags.values == facet_value]
            for facet_name, facet_value in facets_tags_values.items()
        }
