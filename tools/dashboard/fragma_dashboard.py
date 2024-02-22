import pandas as pd

from bokeh.server import server
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import figure, curdoc

# Add some introductory text
text = '<h1 style="text-align: center">Dashboard for fragma</h1>\n'
text += '<h2>How to use</h2>\n'
text += '<ul>\n'
text += '<li>To update the dashboard, refresh this page.</li>\n'
text += '<li>Lines can be hidden by clicking on their labels in the legend.</li>\n'
text += '</ul>\n'
title = Div(text=text)

# Define labels
energies = ["elastic_energy", "fracture_dissipation", "external_work"]
energies_labels = {
        "elastic_energy": "Elastic energy",
        "fracture_dissipation": "Fracture dissipation",
        "external_work": "External work"}
energies_colors = {
        "elastic_energy": "#003162",
        "fracture_dissipation": "#f1443a",
        "external_work": "#007e9f"}
energies_markers = {
        "elastic_energy": "circle",
        "fracture_dissipation": "square",
        "external_work": "triangle"}

# Read the result file
df = pd.read_csv("results/probes.csv")

# Create the data source
energies_data_dict = {energy: df[energy] for energy in energies if energy in df.columns}
energies_data_dict["index"] = df.index
energy_source = ColumnDataSource(data=energies_data_dict)

# Create the energy plot
energy_plot = figure(title="Energies")
# Add line for each energy
for energy in energies:
    if energy in df.columns:
        energy_plot.scatter(
            x = "index",
            y = energy,
            source=energy_source,
            size=10,
            legend_label=energies_labels[energy],
            color=energies_colors[energy],
            marker=energies_markers[energy],
        )
# Set labels
energy_plot.xaxis.axis_label = 'Load step'
energy_plot.yaxis.axis_label = 'Energy (J)'
# Add grid lines
energy_plot.xgrid.grid_line_color = "#cccccc"
energy_plot.ygrid.grid_line_color = "#cccccc"
# Position the legend
energy_plot.add_layout(energy_plot.legend[0], 'above')
# energy_plot.legend.location = "top_left"
# Hide line on click in legend
energy_plot.legend.click_policy="hide"
energy_plot.legend.orientation="horizontal"

# Create the layoutu
layout = column(title, energy_plot)

# Add the layout to the document
curdoc().add_root(layout)

