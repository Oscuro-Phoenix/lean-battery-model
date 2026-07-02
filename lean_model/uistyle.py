"""Shared dark, minimal Plotly styling used across every page of the app."""

PLOT_TEMPLATE = "plotly_dark"
PLOT_PAPER = "#0e1117"
PLOT_GRID = "rgba(255,255,255,0.08)"
MARKER_LINE = "rgba(255,255,255,0.55)"
AXIS_STYLE = dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID,
                  linecolor="rgba(255,255,255,0.25)")

CHEM_COLORS = {
    "cathode": "#ff6f59", "anode": "#4f8cff", "full cell": "#2ecc9a",
    "pseudocapacitor": "#c792ea", "other": "#9aa4b2",
}


def style_fig(fig, legend=None, **layout):
    """Apply the shared dark/minimal layout to a figure in one place."""
    fig.update_layout(
        template=PLOT_TEMPLATE, paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_PAPER,
        font=dict(color="#e6edf3", size=13),
        legend=legend or dict(orientation="h", yanchor="bottom", y=1.02,
                              bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        **layout)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig
