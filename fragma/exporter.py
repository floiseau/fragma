"""
Exporter Module
===============

This module provides functionality to export simulation results.

Classes:
    Exporter: Class for exporting simulation results.
    VTKFieldExporter: Class for exporting field data to VTK files.
    ProbeExporter: Class for exporting probe data to CSV files.
"""
import csv
from pathlib import Path

from dolfinx import io


class Exporter:
    """
    Class for exporting simulation results.

    This class manages exporting simulation results, including field data and probe data.

    Parameters
    ----------
    mesh : dolfinx.Mesh
        The mesh used in the simulation.
    functions_to_export : list of dolfinx.Function
        List of functions to export.
    probes : dict
        Dictionary containing probe information.
    reaction_forces: dict
        Dictionnary containing the value of the reaction forces.

    Attributes
    ----------
    field_exporter : VTKFieldExporter
        Field exporter object.
    scalar_exporter : ScalarExporter
        Scalar exporter object.
    """

    def __init__(self, mesh, functions_to_export, probes, reaction_forces):
        """
        Initialize the Exporter.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh used in the simulation.
        functions_to_export : list of dolfinx.Function
            List of functions to export.
        probes : dict
            Dictionary containing probe information.
        """
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Create the VTKFieldExporter
        self.field_exporter = VTKFieldExporter(
            mesh, functions_to_export, results_folder
        )
        # Create the probe exporter
        self.scalar_exporter = ScalarExporter(probes, reaction_forces, results_folder)

    def export(self, t):
        """
        Export simulation results.

        This method exports simulation results to VTK and CSV formats at a given time.

        Parameters
        ----------
        t : float
            Current time.
        """
        # Run the field exporter
        self.field_exporter.export(t)
        # Run the scalar exporter
        self.scalar_exporter.export(t)

    def end(self):
        """Finalize exporting.

        This method finalizes the export process by closing any open files.
        """
        # End the probe exporter
        self.scalar_exporter.end()
        # End the field exporter
        self.field_exporter.end()


class VTKFieldExporter:
    """
    Class for exporting field data to VTK files.

    This class exports field data to VTK files.

    Parameters
    ----------
    mesh : dolfinx.Mesh
        The mesh representing the domain.
    functions_to_export : list of dolfinx.Function
        List of functions to export.
    results_folder : Path
        Path to the folder where results will be stored.

    Attributes
    ----------
    functions_to_export : list of dolfinx.Function
        List of functions to export.
    files : list of io.VTKFile
        List of VTK files.
    """

    def __init__(self, mesh, functions_to_export, results_folder: Path):
        """
        Initialize the VTKFieldExporter.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
        functions_to_export : list of dolfinx.Function
            List of functions to export.
        results_folder : Path
            Path to the folder where results will be stored.
        """
        print("Warning: Using VTK exporter. This exporter might be slow.")
        # Store the functions to export
        self.functions_to_export = functions_to_export
        # Generate the files
        self.files = []
        for function in functions_to_export:
            # Set the file name
            file_name = results_folder / function.name
            # Create the VTK file
            new_file = io.VTKFile(mesh.comm, file_name.with_suffix(".pvd"), "w")
            # Add the new file to the list
            self.files.append(new_file)

    def export(self, t):
        """
        Export field data.

        This method exports field data to VTK files.

        Parameters
        ----------
        t : float
            Current time.
        """
        # Write the function to the file
        for file, function in zip(self.files, self.functions_to_export):
            # Write the function into the file
            file.write_function(function, t)

    def end(self):
        """
        Finalize the export process.

        This method closes any open files.
        """
        # Close the file
        for file in self.files:
            file.close()


class ScalarExporter:
    """
    Class for exporting scalar data to CSV files.

    This class exports scalar data to CSV files.

    Parameters
    ----------
    probes : dict
        Dictionary containing probes for simulation results.
    reaction_forces: dict
        Dictionary containing the reaction forces.
    results_folder : Path
        Path to the folder where results will be stored.

    Attributes
    ----------
    probes : dict
        Dictionary containing probes for simulation results.
    reaction_forces: dict
        Dictionary containing the reaction forces.
    csv_file : file
        CSV file for storing probe data.
    writer : csv.writer
        CSV writer object.
    """

    def __init__(self, probes, reaction_forces, results_folder: Path):
        """
        Initialize the ScalarExporter.

        Parameters
        ----------
        probes : dict
            Dictionary containing probes for simulation results.
        reaction_forces: dict
            Dictionary containing the reaction forces.
        results_folder : Path
            Path to the folder where results will be stored.
        """
        # Store the probes
        self.probes = probes
        # Store the reaction forces
        self.reaction_forces = reaction_forces
        # Generate the CSV file
        self.csv_file = open(results_folder / "probes.csv", "w")
        # Create the csv writer
        self.writer = csv.writer(self.csv_file)
        # Write the header
        header = []
        # Add the probes
        for func_name, probe in probes.items():
            # Iterate through the probes of the function
            for i, x in enumerate(probe.xs):
                for comp, val in enumerate(probe.vals[i]):
                    # Set the name of the row
                    header.append(f"{func_name} {comp+1} {x}")
        # Add the reaction forces
        for name, reaction_force in reaction_forces.items():
            header.append(name)
        # Write the header
        self.writer.writerow(header)

    def export(self, t: float):
        """
        Export probe data to CSV files.

        This method exports probe data to CSV files for a given time.

        Parameters
        ----------
        t : float
            Current time.
        """
        # Write the row
        row = []
        for func_name, probe in self.probes.items():
            # Iterate through the probes of the function
            for i, _ in enumerate(probe.xs):
                for val in probe.vals[i]:
                    # Add the value to the row
                    row.append(val)
        # Add the reaction forces
        for name, reaction_force in self.reaction_forces.items():
            row.append(reaction_force)
        # Write the row
        self.writer.writerow(row)
        # Flush the results
        self.csv_file.flush()

    def end(self):
        """
        Finalize the export process.

        This method closes any open files.
        """
        self.csv_file.close()
