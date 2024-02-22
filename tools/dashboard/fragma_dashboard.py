import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Load data using pandas
df = pd.read_csv("results/probes.csv")
df["Load step"] = df.index

# Initialize Dash app
app = dash.Dash(__name__)

# Add some introductory text
intro_text = """
# Dashboard for fragma

## How to use

- To update the dashboard, refresh this page.
- Select the quantities along the x and y axis using the dropdown menu below the plot.
- Lines can be hidden by clicking on their labels in the legend.

## Custom plot
"""

# Define layout
app.layout = html.Div(
    [
        dcc.Markdown(intro_text),
        dcc.Graph(id="custom-plot"),
        html.Div(
            [
                html.Label("Choose quantity on x axis:"),
                dcc.Dropdown(
                    id="x-select",
                    options=[{"label": col, "value": col} for col in df.columns],
                    value=df.columns[-1],
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Choose quantity on y axis:"),
                dcc.Dropdown(
                    id="y-select",
                    options=[{"label": col, "value": col} for col in df.columns],
                    value=[df.columns[0]],
                    multi=True,
                ),
            ]
        ),
    ],
    className="ag-theme-alpine-dark",
)


# Define callback to update scatter plot
@app.callback(
    Output("custom-plot", "figure"),
    [Input("x-select", "value"), Input("y-select", "value")],
)
def update_scatter_plot(x_name, y_names):
    # Display an empty plot if nothing is selected
    if not y_names:
        return px.scatter()
    # Reset the figure
    fig = px.scatter()
    # Add the different lines
    for y_name in y_names:
        print(x_name, y_name)
        fig.add_scatter(
            x=df[x_name],
            y=df[y_name],
            mode="lines+markers",
            name=y_name,
            line=dict(width=1),
        )
    # Update the layout
    fig.update_layout(xaxis_title=x_name, yaxis_title="")
    # Show the grid
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgrey",
        ticks="outside",
        showline=True,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        ticks="outside",
        showline=True,
        linecolor="black",
        mirror=True,
    )
    # Return the figure
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
