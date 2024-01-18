import csv
from pathlib import Path

from dolfinx import io

class Exporter:

    def __init__(self, mesh, functions_to_export, probes):
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Create the VTKFieldExporter
        self.field_exporter = VTKFieldExporter(mesh, functions_to_export, results_folder)
        # Create the probe exporter
        self.probe_exporter = ProbeExporter(probes, results_folder)


    def export(self, t):
        # Run the field exporter
        self.field_exporter.export(t)
        # Run the probe exporter
        self.probe_exporter.export(t)

    def end(self):
        # End the probe exporter
        self.probe_exporter.end()
        # End the field exporter
        self.field_exporter.end()


class VTKFieldExporter:
    def __init__(self, mesh, functions_to_export, results_folder: Path):
        print("Warning: Using VTK exporter. This exporter might be slow.")
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

    def end(self):
        # Close the file
        for file in self.files:
            file.close()

class ProbeExporter:
    def __init__(self, probes, results_folder: Path):
        # Store the probes
        self.probes = probes
        # Generate the CSV file
        self.csv_file = open(results_folder/"probes.csv", "w") 
        # Create the csv writer
        self.writer = csv.writer(self.csv_file)
        # Write the header
        header = []
        for func_name, probe in probes.items():
            # Iterate through the probes of the function
            for i, x in enumerate(probe.xs):
                for comp, val in enumerate(probe.vals[i]):
                    # Set the name of the row
                    header.append(f"{func_name} {comp+1} {x}")
        # Write the header
        self.writer.writerow(header)

    def export(self, t: float):
        # Write the header
        row = []
        for func_name, probe in self.probes.items():
            # Iterate through the probes of the function
            for i, _ in enumerate(probe.xs):
                for val in probe.vals[i]:
                    # Add the value to the row
                    row.append(val)
        # Write the row
        self.writer.writerow(row)

    def end(self):
        self.csv_file.close()
