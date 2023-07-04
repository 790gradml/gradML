import numpy as np
import plotly.graph_objects as go
import colors as c

n = 100
min_x = -3
max_x = 3
min_y = -60
max_y = 40

# Generate data
f = lambda alpha, x: alpha * x * x + x + 3 * x
alphas = np.linspace(min_x * 2, max_x * 2, 7)
x = np.random.randn(n) - 0.43
xbar = np.average(x)

y = {}
ybar = {}
f_of_x_bar = {}
for alpha in alphas:
    this_y = f(alpha, x)
    f_of_x_bar[str(alpha)] = f(alpha, xbar)
    y[str(alpha)] = this_y
    ybar[str(alpha)] = np.average(this_y)


# Create figure and scatter plot


def vertical_line(x, y):
    return go.Scatter(
        x=[x, x],  # x-coordinates of the vertical line
        y=[min_y, y],  # y-coordinates of the vertical line
        mode="lines",
        name="Vertical Line",
        marker=dict(color=c.BLUE),
        showlegend=False,
    )


# Create the horizontal line trace
def horizontal_line(x, y):
    return go.Scatter(
        x=[min_x, x],  # x-coordinates of the horizontal line
        y=[y, y],  # y-coordinates of the horizontal line
        mode="lines",
        name="Horizontal Line",
        marker=dict(color=c.PURPLE),
        showlegend=False,
    )


def scatter_on_f(i=alphas[0]):
    return [
        go.Scatter(
            x=x,
            y=y[str(i)],
            mode="markers",
            marker=dict(size=4, showscale=False),
            hovertemplate="x: %{x}<br>y: %{y}",
            showlegend=False,
        ),
        horizontal_line(xbar, f_of_x_bar[str(i)]),
        vertical_line(xbar, f_of_x_bar[str(i)]),
    ]


subplot_x_axis = [
    go.Scatter(
        x=x,
        y=np.zeros(n),
        mode="markers",
        xaxis="x",
        yaxis="y2",
        marker=dict(size=4, showscale=False),
        showlegend=False,
    ),
    go.Scatter(
        x=np.array(xbar),
        y=np.zeros(1),
        xaxis="x",
        yaxis="y2",
        mode="markers",
        marker=dict(size=12, symbol="star", color=c.BLUE),
        hovertemplate="the mean of x samples: %{xbar}",
        name="$\mathbb{E}[x]$",
    ),
]


def subplot_y_axis(i=alphas[0]):
    return [
        go.Scatter(
            x=np.zeros(n),
            y=y[str(i)],
            xaxis="x2",
            yaxis="y",
            mode="markers",
            marker=dict(size=4, showscale=False),
            showlegend=False,
        ),
        go.Scatter(
            x=np.zeros(1),
            y=[ybar[str(i)]],
            xaxis="x2",
            yaxis="y",
            mode="markers",
            marker=dict(size=12, symbol="square", color=c.RED),
            hovertemplate="the mean of f(x): %{ybar}",
            name="$\mathbb{E}[f(x)]$",
        ),
        go.Scatter(
            x=np.zeros(1),
            y=[f_of_x_bar[str(i)]],
            xaxis="x2",
            yaxis="y",
            mode="markers",
            marker=dict(size=12, symbol="star", color=c.PURPLE),
            hovertemplate="f of the mean of x: %{f_of_x_bar[0]}",
            name="$f(\mathbb{E}[x])$",
        ),
    ]


fig = go.Figure(
    data=scatter_on_f() + subplot_x_axis + subplot_y_axis(),
    layout=go.Layout(
        # title="Jensen's Inequality",
        # xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[min_y, max_y]),
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
                currentvalue={"prefix": "α: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=(i),
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
            data=scatter_on_f(i=i) + subplot_x_axis + subplot_y_axis(i=i),
        )
        for i in alphas
    ],
)

legend = dict(orientation="v", x=0.16, y=0.16)
# Show the figure
fig.update_layout(
    title="Jensen's Inequality",
    xaxis=dict(
        title="Sampled $x$",
        domain=[0.15, 1],
    ),
    yaxis=dict(title="$f(x)$", domain=[0.15, 1], tickformat="$%s$"),
    xaxis2=dict(title=None, domain=[0, 0.03], range=(-0.1, 0.1), showticklabels=False),
    yaxis2=dict(title=None, domain=[0, 0.03], range=(-0.1, 0.1), showticklabels=False),
    showlegend=True,
    legend=legend,
)


import os

script_folder = os.path.dirname(os.path.abspath(__file__))
fig.write_html(
    script_folder + "/../Jensen.html",
    include_mathjax="/assets/js/MathJax/MathJax.js",
)
