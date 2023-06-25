import plotly.graph_objects as go
import numpy as np

# TODO: use proper LaTeX instead of Unicode hack

# from pandas import DataFrame
# Create figure
fig = go.Figure()
r_max = [1,1.5]
x = np.linspace(-r_max[0], r_max[0], 60)
y = np.linspace(-r_max[1], r_max[1], 60)
X, Y = np.meshgrid(x, y)
def quadratic(X,Y,Q, q):
    Z = Q[0,0]*X**2+Q[1,1]*Y**2+2*Q[0,1]*X*Y + q[0]*X + q[1]*Y
    return Z

Qs = [np.array([[4,2],[2,5]]),
      np.array([[1,2],[2,4]]),
      np.array([[1,1],[1,1]]),
      np.array([[-1,2],[2,4]])]

qs = [np.array([1,2]),
      np.array([0,0]),
      np.array([-10,0]),
      np.array([0,0])]
titles = ["f(x₁,x₂)=4x₁²+5x₂²+4x₁x₂+x₁+2x₂, f strictly convex",
          "f(x₁,x₂)=x₁²+4x₂²+4x₁x₂, f convex but not strictly convex",
          "f(x₁,x₂)=x₁²+x₂²+2x₁x₂-10x₁, f convex but not strictly convex",
          "f(x₁,x₂)=-x₁²+4x₂²+4x₁x₂, f non-convex",
          ]

labels = ["f strictly convex",
          "f convex\n but not strictly convex",
          "f convex but not strictly convex",
          "f non-convex",
          ]

a=np.array([-0.0312])
b = np.array([-0.1875])
fig.add_trace(go.Scatter3d(x=a,y=b, z=quadratic(a,b,Qs[0],qs[0])))
# Add traces, one for each slider step
pi = np.pi
cos = np.cos
sin = np.sin
phi = np.linspace(0, 2*pi)
theta = np.linspace(-pi/2, pi/2)
phi, theta=np.meshgrid(phi, theta)

for (Q,q) in zip(Qs,qs):
    Z = quadratic(X,Y, Q, q)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, colorscale='Sunset', opacity=1,
    )
    )

for i in fig.data:
    i.visible = False
fig.data[0].visible = True
fig.data[1].visible = True

# Create and add slider
steps = []
for (j,i) in enumerate(Qs):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": titles[j]}],# layout attribute
        label = str(i)
    )
    if j ==0:
        step["args"][0]["visible"][j] = True
    step["args"][0]["visible"][j+1] = True
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Q: "},
    pad={"t": 50},
    steps=steps,
)]

fig.update_layout(
    sliders=sliders,
    autosize=True,
    title = titles[0],
    title_font_size=20,
    showlegend=False,
        scene=dict(
            xaxis=dict(title='x₁'),
            yaxis=dict(title=r'x₂'),
            zaxis=dict(title=r'f(x₁,x₂)', nticks=4)
        ),
)

fig.update_layout()
fig.show()
fig.write_html("../QuadraticFun.html")
