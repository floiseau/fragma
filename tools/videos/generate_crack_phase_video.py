from pathlib import Path
import pyvista as pv

# Create a video directory
Path("videos").mkdir(exist_ok=True)
# Read data
reader = pv.get_reader("results/CrackPhase.pvd")
# Get time steps
ts = reader.time_values
# Create a plotter
p = pv.Plotter(off_screen=True)
# Open a gif
p.open_gif("videos/CrackPhase.gif", fps=len(ts) / 5)

for t in ts:
    # Reset the plotter
    p.clear()
    # Set the time value
    reader.set_active_time_value(t)
    # Read data
    crack_data = reader.read()
    # Add the mesh to the plot
    p.add_mesh(crack_data, scalars="CrackPhase")
    p.view_xy()
    # Add frame to the gif
    p.write_frame()
p.close()
