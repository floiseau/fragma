from mpi4py import MPI

from dolfinx import io


class Domain:
    def __init__(self, mesh_pars, dim):
        print("\n████ READING THE MESH")
        # Read the mesh from GMSH
        print("Mesh reading output:")
        msh_file = mesh_pars["msh_file"]
        self.mesh, self.cell_tags, self.facet_tags = io.gmshio.read_from_msh(
            msh_file, MPI.COMM_WORLD, gdim=dim
        )
        # Locate the physical groups
        self.__locate_physical_groups(mesh_pars["physical_groups"])

    def __locate_physical_groups(self, facets_tags_values):
        # Get the facets indices
        self.boundary_facets = {}
        for facet_name, facet_value in facets_tags_values.items():
            self.boundary_facets[facet_name] = self.facet_tags.indices[
                self.facet_tags.values == facet_value
            ]
