from pathlib import Path
import pyvista as pv

# Create a video directory
Path("videos").mkdir(exist_ok=True)
# Read data
reader = pv.get_reader('results/Displacement.pvd')
# Get time steps
ts = reader.time_values
# Create a plotter
p = pv.Plotter(off_screen=True)
# Open a gif
p.open_gif("videos/Displacement.gif", fps=len(ts)/5)


for t in reader.time_values:
    # Reset the plotter
    p.clear()
    # Set the time value
    reader.set_active_time_value(t)
    # Read data
    disp_data = reader.read()
    # Warp by vector
    data = disp_data.get("Block-00").warp_by_vector(factor=10)
    # Add the mesh to the plot
    p.add_mesh(data, scalars="Displacement")
    p.view_xy()
    # Add frame to the gif
    p.write_frame()
p.close()
