import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from complete_graph import all_pairs, is_connected

COLORS=['red','green','blue','black']

def random_pairs(nodes, p):
    for edge in all_pairs(nodes):
        if flip(p):
            #print(edge)
            yield edge

def flip(p):
    return np.random.random() < p

def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)

    G.add_edges_from(random_pairs(nodes, p))
    return G

def prob_connected(n, p, iters=100):
    tf = [is_connected(make_random_graph(n, p))
            for i in range(iters)]
    return np.mean(tf)

def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen



if __name__ == '__main__':
    random_graph = make_random_graph(10, 0.3)
    nx.draw_circular(random_graph,
                     node_color=COLORS[0],
                     node_size=1000,
                     with_labels=True)
    print(prob_connected(10, 0.23, iters=10000))
    n = 10
    ps = np.logspace(-2.5, 0, 11)
    ys = [prob_connected(n, p) for p in ps]
    print(ys)
    s = reachable_nodes(random_graph, 0)
    print(s)

    plt.show()

