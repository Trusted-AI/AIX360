
import numpy as np
from itertools import product
from typing import Set, Optional, List, Tuple
from matplotlib.axes import Axes

def plot_transport_plan_graph(
    doc_a: List[Tuple[str, float]], 
    doc_b: List[Tuple[str, float]], 
    X: np.ndarray, 
    ax: Axes,
    threshold: float = 1e-4,
    highlight_edges: Set[Tuple[int,int]] = set([]),
    highlight_nodes: Optional[Tuple[List[str], List[str]]] = None,
    **kwargs,
):
    """
    Utility function to plot the transport plan.
    """

    from matplotlib import cm
    from matplotlib.colors import Normalize
    import networkx as nx 

    BLUE = (0.,0.,1.,1.) # RGBA for blue

    _a, _ = zip(*doc_a)
    _b, _ = zip(*doc_b)
    m, n = len(_a), len(_b)

    _a = [(0,i,x) for i,x in enumerate(_a)]
    _b = [(1,i,x) for i,x in enumerate(_b)]

    G = nx.Graph()

    # for node in 
    [G.add_node(x) for x in _a]
    [G.add_node(x) for x in _b]

    edge_values=[]
    edge_colors=[]
    _cmap=cm.get_cmap('binary')
    for (i, x), (j,y) in product(
        enumerate(_a),
        enumerate(_b),
    ):
        if X[i,j] > threshold:
            G.add_edge(x, y)
            edge_values.append(
                ((i, j), X[i,j]),
            )

    _, _values = zip(*edge_values)

    color_normal = Normalize(
        vmin=kwargs.get("vmin", min(_values)),
        vmax=kwargs.get("vmax", max(_values)),
    )
    edge_colors = [
        _cmap(color_normal(x)) if i not in highlight_edges
        else BLUE
        for i,x in edge_values
    ]

    if highlight_nodes is not None:
        _hna, _hnb = highlight_nodes
        node_color = _hna + _hnb
    else:
        node_color = "#ffffff"

    pos = {
        **{
            (b,i,x): [i-(m-1)/2, 1]
            for b, i, x in _a
        },
        **{
            (b,i,x): [i-(n-1)/2, -1]
            for b, i, x in _b
        }
    }

    nx.draw_networkx(
        G,
        pos=pos,
        arrows=False,
        ax=ax,
        labels={k:k[2] for k in pos.keys()},
        edge_color=edge_colors,
        font_family='monospace',
        font_size=16,
        node_size=500,
        node_color=node_color,
    )
