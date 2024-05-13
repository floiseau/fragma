from pathlib import Path
import pyvista as pv

# Create a video directory
Path("videos").mkdir(exist_ok=True)
# Read data
displ_reader = pv.get_reader('results/Displacement.pvd')
crack_reader = pv.get_reader('results/CrackPhase.pvd')
# Get time steps
ts = displ_reader.time_values
# Create a plotter
p = pv.Plotter(off_screen=True)
# Open a gif
p.open_gif("videos/CrackPhase_warped.gif", fps=len(ts)/(5*5))

for t in ts:
    if not t % 5 == 0:
        continue
    # Reset the plotter
    p.clear()
    # Set the time value
    displ_reader.set_active_time_value(t)
    crack_reader.set_active_time_value(t)
    # Read displacement data
    displ_data = displ_reader.read().get("Block-00")
    # Warp by vector
    data = displ_data.warp_by_vector(factor=10)
    # Add crack phase data
    crack_data = crack_reader.read().get("Block-00")
    data.point_data["CrackPhase"] = crack_data.point_data["CrackPhase"]
    # Add a threshold
    data = data.threshold(0.99, method="lower")
    # Add the mesh to the plot
    p.add_mesh(data, scalars="CrackPhase")
    p.view_xy()
    # Add frame to the gif
    p.write_frame()
p.close()
