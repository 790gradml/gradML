import numpy as np
import plotly.graph_objects as go

# Generate random data
f = lambda alpha, x: alpha * x * x + x + 3 * x

# Create NumPy arrays
alphas = np.linspace(-3, 3, 9)
x = np.random.randn(200) + 1
y = {}
for alpha in alphas:
    # mean_of_x = np.average(x)
    y[str(alpha)] = f(alpha, x)
# Create figure and scatter plot
fig = go.Figure(
    data=go.Scatter(
        x=x,
        y=y[str(alphas[0])],
        mode="markers",
        marker=dict(size=10, colorscale="Viridis", showscale=True),
        hovertemplate="x: %{x}<br>y: %{y}<br>scalar: %{marker.color}",
    ),
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
                currentvalue={"prefix": "Alpha: "},
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
                )
            ],
        )
        for i in alphas
    ],
)

# Show the figure
fig.update_layout(title_text="Jensen's Inequality")
fig.show()

import os

script_folder = os.path.dirname(os.path.abspath(__file__))
fig.write_html(script_folder + "/../Jensen.html")
