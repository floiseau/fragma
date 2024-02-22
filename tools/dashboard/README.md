# Dashboard for fragma

## Description
This tool generates an HTML dashboard using the dash module (which uses plotly) to monitor fragma simulations.

## Dependencies
To use the dashboard, ensure that the dash package is installed in the `fragma` Conda environment. You can install it using the following command:
```bash
conda activate fragma
conda install dash
```

## Usage
1. Activate the fragma environment (including the optional dash dependency) using the following command:
```bash
conda activate fragma
```
2. Start the Dash server by running the following command in the simulation directory:
```bash
python /path/to/fragma/tools/dashboard/tools/dashboard/fragma_dashboard.py
```
3. This command should start the Dash dashboard and display the following message in the command line:
```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'test_plotly'
 * Debug mode
```
4. Open a web browser and navigate to the following address to access the dashboard: http://127.0.0.1:8050/.

## Notes
- The dashboard provides various usage tips and instructions within the dashboard interface.
