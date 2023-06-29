import numpy as np
import plotly.graph_objects as go

n = 200
# Generate random data
f = lambda alpha, x: alpha * x * x + x + 3 * x

# Create NumPy arrays
alphas = np.linspace(-3, 3, 9)
x = np.random.randn(n) + 1
xbar = np.average(x)
f_of_x_bar = f(alpha, xbar)
y = {}
ybar = {}
for alpha in alphas:
    this_y = f(alpha, x)
    y[str(alpha)] = this_y
    ybar[str(alpha)] = np.average(this_y)
# Create figure and scatter plot
fig = go.Figure(
    data=[
        go.Scatter(
            x=x,
            y=y[str(alphas[0])],
            mode="markers",
            marker=dict(size=10, colorscale="Viridis", showscale=False),
            hovertemplate="x: %{x}<br>y: %{y}",
        ),
        go.Scatter(
            x=x,
            y=np.zeros(n),
            mode="markers",
            xaxis="x",
            yaxis="y2",
        ),
        go.Scatter(
            x=np.zeros(n),
            y=y[str(alphas[0])],
            xaxis="x2",
            yaxis="y",
        ),
    ],
    layout=go.Layout(
        # title="Jensen's Inequality",
        # xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-30, 30]),
        # updatemenus=[
        #     dict(
        #         type="buttons",
        #         buttons=[
        #             dict(
        #                 label="Play",
        #                 method="animate",
        #                 args=[
        #                     None,
        #                     {
        #                         "frame": {"duration": 500, "redraw": True},
        #                         "fromcurrent": True,
        #                         "transition": {"duration": 0},
        #                     },
        #                 ],
        #                 # Add 'args2' to synchronize the slider with the animation frames
        #                 args2=[
        #                     None,
        #                     {
        #                         "frame": {"duration": 500, "redraw": True},
        #                         "mode": "immediate",
        #                         "transition": {"duration": 0},
        #                     },
        #                 ],
        #             )
        #         ],
        #     )
        # ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": r"$\alpha$: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=str(i),
                        method="animate",
                        args=[
                            [f"frame{i}"],
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    )
                    for i in alphas
                ],
            )
        ],
    ),
    frames=[
        go.Frame(
            name=f"frame{i}",
            data=[
                go.Scatter(
                    x=x,
                    y=y[str(i)],
                    mode="markers",
                    marker=dict(
                        size=10,
                    ),
                ),
                go.Scatter(
                    x=x,
                    y=np.zeros(n),
                    mode="markers",
                    name="Subplot 1",
                    xaxis="x",
                    yaxis="y2",
                ),
                go.Scatter(
                    x=np.zeros(n),
                    y=y[str(i)],
                    mode="markers",
                    xaxis="x2",
                    yaxis="y",
                ),
            ],
        )
        for i in alphas
    ],
)

# Show the figure
fig.update_layout(
    title="Jensen's Inequality",
    xaxis=dict(title="Sampled x", domain=[0, 0.95]),
    yaxis=dict(title=r"$f(x)$", domain=[0.1, 1], tickformat="$%s$"),
    xaxis2=dict(
        title=None, domain=[1 - 0.03, 1], range=(-0.1, 0.1), showticklabels=False
    ),
    yaxis2=dict(title=None, domain=[0, 0.03], range=(-0.1, 0.1), showticklabels=False),
    showlegend=False,
)


# Create the figure
# fig.show()

import os

script_folder = os.path.dirname(os.path.abspath(__file__))
fig.write_html(script_folder + "/../Jensen.html")
