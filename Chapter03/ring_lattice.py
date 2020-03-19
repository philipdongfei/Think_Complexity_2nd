import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from collections import deque

#from decorate import *
import decorator
from savefig import *

np.random.seed(17)
from warnings import simplefilter
import matplotlib.cbook
simplefilter('ignore', matplotlib.cbook.mplDeprecation)

def adjacent_edges(nodes, halfk):
    n = len(nodes)
    for i, u in enumerate(nodes):
        for j in range(i+1, i+halfk+1):
            v = nodes[j % n]
            yield u, v

def make_ring_lattice(n, k):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(adjacent_edges(nodes, k//2))
    return G
def rewire(G, p):
    nodes = set(G)
    for u, v in G.edges():
        if flip(p):
            choices = nodes - {u} - set(G[u])
            new_v = np.random.choice(list(choices))
            G.remove_edge(u, v)
            G.add_edge(u, new_v)

def make_ws_graph(n, k, p):
    '''Makes a Watts-Strogatz graph.
    n: number of nodes
    k: degree of each node
    p: probability of rewiring an edge
    '''
    ws = make_ring_lattice(n, k)
    rewire(ws, p)
    return ws
def flip(p):
    """Returns True with probability p."""
    return np.random.random() < p
def node_clustering(G, u):
    neighbors = G[u]
    k = len(neighbors)
    if k < 2:
        return np.nan
    possible = k * (k-1) / 2
    exist = 0
    for v, w in all_pairs(neighbors):
        if G.has_edge(v, w):
            exist += 1
    return exist / possible

def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:
                    yield u, v

def clustering_coefficient(G):
    cu = [node_clustering(G, node) for node in G]
    return np.nanmean(cu)
def clustering_coefficient_new(G):
    cu = []
    for node in G:
        cu.append(node_clustering(G, node))
    return np.nanmean(cu)

def path_lengths(G):
    length_iter = nx.shortest_path_length(G)
    for source, dist_map in length_iter:
        for dest, dist in dist_map.items():
            yield dist
    """
    length_map = nx.shortest_path_length(G)
    """
    #lengths = [length_map[u][v] for u, v in all_pairs(G)]
    #return lengths

def characteristic_path_length(G):
    return np.mean(list(path_lengths(G)))

def run_one_graph(n, k, p):
    ws = make_ws_graph(n, k, p)
    mpl = characteristic_path_length(ws)
    cc = clustering_coefficient(ws)
    return mpl, cc

def run_experiment(ps, n=1000, k=10, iters=20):
    """Computes stats for WS graphs with a range of `p`.
    ps: sequence of `p` to try
    n: number of nodes
    k: degree of each node
    iters: number of times to run for each `p`
    return:
    """
    res=[]
    for p in ps:
        print(p)
        t = [run_one_graph(n, k, p) for _ in range(iters)]
        means = np.array(t).mean(axis=0)
        print(means)
        res.append(means)
    return np.array(res)

def reachable_nodes(G, start):
    """DFS"""
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen
def reachable_nodes_bfs(G, start):
    """BFS"""
    seen = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in seen:
            seen.add(node)
            queue.extend(G.neighbors(node))
    return seen
def shortest_path_dijkstra(G, source):
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        #print('dijkstra node: ', node)
        new_dist = dist[node] + 1
        # find all neighbors of node that are not already in dist
        neighbors = set(G[node]).difference(dist)
        for n in neighbors:
            dist[n] = new_dist
        queue.extend(neighbors)
    return dist

if __name__ == '__main__':
    nodes = range(3)
    for edge in adjacent_edges(nodes, 1):
        print(edge)
    n = 10
    k = 4
    lattice = make_ring_lattice(n, k)
    nx.draw_circular(lattice,
                     node_color='C0',
                     node_size=1000,
                     with_labels=True)
    #savefig_png('figs/chap03-1')
    ws = make_ws_graph(10, 4, 0.2)
    nx.draw_circular(ws,
                     node_color='C1',
                     node_size=1000,
                     with_labels=True)
    print(len(lattice.edges()), len(ws.edges()))
    print('node clustering: ', node_clustering(lattice, 1))
    print('coefficient: ' , clustering_coefficient(lattice))
    print('new coefficient: ' , clustering_coefficient_new(lattice))
    n = 10
    k = 4
    ns = 100
    '''

    plt.subplot(1,3,1)
    ws = make_ws_graph(n, k, 0)
    nx.draw_circular(ws, node_size=ns)
    plt.axis('equal')

    plt.subplot(1,3,2)
    ws = make_ws_graph(n, k, 0.2)
    nx.draw_circular(ws, node_size=ns)
    plt.axis('equal')

    plt.subplot(1,3,3)
    ws = make_ws_graph(n, k, 1.0)
    nx.draw_circular(ws, node_size=ns)
    plt.axis('equal')
    '''

    lattice = make_ring_lattice(10, 4)
    print(node_clustering(lattice, 1))
    print(clustering_coefficient(lattice))
    print(clustering_coefficient_new(lattice))

    lattice = make_ring_lattice(1000,10)
    print('(1000,10)characteristic path length: ', characteristic_path_length(lattice))

    lattice = make_ring_lattice(10,4)
    print('(10,4)characteristic path length: ', characteristic_path_length(lattice))

    print('run one graph: 1000, 10, 0.01: ')
    run_one_graph(1000, 10, 0.01)

    ps = np.logspace(-4, 0, 9)
    print('ps: ', ps)

    res = run_experiment(ps)
    print('res:')
    print(res)
    L, C = np.transpose(res)
    print('L:', L)
    print('C:', C)
    L /= L[0]
    C /= C[0]
    plt.plot(ps, C, 's-', linewidth=1, label='c(p) / C(0)')
    plt.plot(ps, L, 'o-', linewidth=1, label='L(p) / L(0)')
    '''

    decorate(xlabel='Rewiring probability (p)', xscale='log',
            title='Normalized clustering coefficient and path length',
            xlim=[0.00009,1.1], ylim=[-0.01,1.01])

    '''
    lattice = make_ring_lattice(10, 4)
    nx.draw_circular(lattice,
                    node_color='C2',
                    node_size=1000,
                    with_labels=True)

    seen = reachable_nodes(lattice, 0)
    print('dfs seen: ', seen)
    seen = reachable_nodes_bfs(lattice,0)
    print('bfs seen: ', seen)
    lattice = make_ring_lattice(10, 4)
    nx.draw_circular(lattice,
                    node_color='C3',
                    node_size=1000,
                    with_labels=True)
    d1 = shortest_path_dijkstra(lattice,0)
    print('d1: ', d1)
    d2 = nx.shortest_path_length(lattice, 0)
    print('d2: ', d2)
    print(d1 == d2)
    print('max diameter: ', max(d1.values()))


    print('n=1000, k=10:')
    lattice = make_ring_lattice(1000, 10)
    nx.draw_circular(lattice,
                    node_color='C4',
                    node_size=1000,
                    with_labels=True)
    d1 = shortest_path_dijkstra(lattice,0)
    #print('d1: ', d1)
    d2 = nx.shortest_path_length(lattice, 0)
    #print('d2: ', d2)
    print(d1 == d2)



    #plt.show()


