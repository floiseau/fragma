# Dashboard for fragma

## Description
This tool is using the `bokeh` module to generate an HTML dashboard to monitor `fragma` simulations.

## How to use
To use this dashboard, one need to activate `fragma` environment (with the optional `bokeh` dependency) and start the `bokeh` server.
This must be done at the root of the simulation directory.
```shell
conda activate fragma
bokeh serve --show /path/to/fragma/tools/dashboard/fragma_dashboard.py
```
This command should automatically open the dashboard in your default web browser.
You can also open using the address shown in the terminal, as show below.
```shell
2024-02-22 11:42:18,382 Bokeh app running at: http://localhost:5006/fragma_dashboard
```
Different usage tips are shown in the dashboard.
