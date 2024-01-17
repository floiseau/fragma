from pathlib import Path

from dolfinx import io


class XDMFExporter:
    def __init__(self, mesh, functions_to_export):
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Set the name of the exported file
        filename = results_folder / "results"
        # Store the functions to export
        self.functions_to_export = functions_to_export
        # Open the file
        self.file = io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w")
        # Export the mesh
        self.file.write_mesh(mesh)

    def export(self, t):
        for function in self.functions_to_export:
            self.file.write_function(function, t)

    def end_export(self):
        # Close the file
        self.file.close()


class VTXExporter:
    """Export the results in VTX format.

    WARNING: This does not work in dolfinx 0.7.2.
    """

    def __init__(self, mesh, functions_to_export):
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Store the functions to export
        self.functions_to_export = functions_to_export
        # Open the file
        self.files = []
        for function in functions_to_export:
            # Create file name
            file_name = results_folder / function.name
            # Create the VTX file
            new_file = io.VTXWriter(
                mesh.comm, file_name.with_suffix(".bp"), [function], engine="BP4"
            )
            # Add the new file to the file list
            self.files.append(new_file)

    def export(self, t):
        for file in self.files:
            file.write(t)

    def end_export(self):
        # Close the file
        for file in self.files:
            file.close()


class VTKExporter:
    def __init__(self, mesh, functions_to_export):
        print("Warning: Using VTK exporter. This exporter might be slow.")
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Store the functions to export
        self.functions_to_export = functions_to_export
        # Generate the files
        self.files = []
        for function in functions_to_export:
            # Set the file name
            file_name = results_folder / function.name
            # Create the VTK file
            new_file = io.VTKFile(mesh.comm, file_name.with_suffix(".vtk"), "w")
            # Export the mesh
            new_file.write_mesh(mesh)
            # Add the new file to the list
            self.files.append(new_file)

    def export(self, t):
        # Write the function to the file
        for file, function in zip(self.files, self.functions_to_export):
            # Write the function into the file
            file.write_function(function, t)

    def end_export(self):
        # Close the file
        for file in self.files:
            file.close()
