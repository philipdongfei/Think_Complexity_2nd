import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
#from utils import decorate, savefig
from savefig import savefig

# I set the random seed so the notebook
# produces the same results every time.
np.random.seed(17)

# TODO: remove this when NetworkX is fixed
from warnings import simplefilter
import matplotlib.cbook
simplefilter('ignore', matplotlib.cbook.mplDeprecation)

# node colors for drawing networks
colors = sns.color_palette('pastel', 5)
# sns.palplot(colors)
sns.set_palette(colors)

# Directed graph
G = nx.DiGraph()
# add nodes
G.add_node('Alice')
G.add_node('Bob')
G.add_node('Chuck')
list(G.nodes())

# add edges
G.add_edge('Alice', 'Bob')
G.add_edge('Alice', 'Chuck')
G.add_edge('Bob', 'Alice')
G.add_edge('Bob', 'Chuck')
list(G.edges())

# draw the graph
nx.draw_circular(G,
                node_color='C0',
                node_size=2000,
                with_labels=True)
plt.axis('equal')
#plt.show()
plt.savefig('figs/chap02-1')
#Exercise: Add another node and a few more edges and draw the graph again

G.add_node('Dick')
G.add_edge('Bob', 'Dick')
G.add_edge('Dick', 'Alice')
nx.draw_circular(G,
                node_color='C0',
                node_size=2000,
                with_labels=True)
plt.axis('equal')
plt.show()

# Undirected graph
positions = dict(Albany=(-74, 43),
                Boston=(-74, 42),
                NYC=(-74, 41),
                Philly=(-75, 40))
positions['Albany']

# add nodes
G = nx.Graph()
G.add_nodes_from(positions)
G.nodes()

# maps from pairs of cities
drive_times = {('Albany', 'Boston'): 3,
               ('Albany', 'NYC'): 4,
               ('Boston', 'NYC'): 4,
               ('NYC', 'Philly'): 2}
# add edges
G.add_edges_from(drive_times)
G.edges()

# Draw
nx.draw(G, positions,
        node_color='C1',
        node_shape='s',
        node_size=2500,
        with_labels=True)
nx.draw_networkx_edge_labels(G, positions,
                            edge_labels=drive_times)
plt.axis('equal')
savefig('figs/chap02-2')
# Exercise: Add another city and at least one edge.


# Complete graph
def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v
def make_complete_graph(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G
def make_complete_digraph(n):
    G = nx.DiGraph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G
# complete graph with 10 nodes
complete = make_complete_graph(10)
complete.number_of_nodes()

# looks like
nx.draw_circular(complete,
                 node_color='C2',
                 node_size=1000,
                 with_labels=True)
savefig('figs/chap02-3')

# neighbors method the neighbors for a given node.
list(complete.neighbors(0))
# Exercise: Make and draw complete directed graph with 5 nodes
comp5 = make_complete_digraph(5)
comp5.number_of_nodes()
nx.draw_circular(complete,
                 node_color='C2',
                 node_size=1000,
                 with_labels=True)
plt.show()

# Random graphs
def flip(p):
    return np.random.random() < p
# random_pairs is a generator function that enumerates all possible pairs of nodes and yields each one with probability p
def random_pairs(nodes, p):
    for edge in all_pairs(nodes):
        if flip(p):
            yield edge
# make_random_graph makes an ER graph where the probability of an edge between each pair of nodes is p.
def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes, p))
    return G

# n=10 and p=0.3
np.random.seed(10)
random_graph = make_random_graph(10, 0.3)
len(random_graph.edges())
# looks like
nx.draw_circular(random_graph,
                 node_color='C3',
                 node_size=1000,
                 with_labels=True)
savefig('figs/chap02-4')

# Connectivity
# check whether a graph is connected, we'll start by finding all nodes
# that can be reached, starting with a given node
def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen
# in the complete graph, starting from node 0, we can reach all nodes
reachable_nodes(complete, 0)

# in the random graph we generated, we can also reach all nodes(but that's not always true)
reachable_nodes(random_graph, 0)

# check whether a graph is connected
def is_connected(G):
    start = next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)
is_connected(complete)
# But if we generate a random graph with a low value of p, it's not
random_graph = make_random_graph(10, 0.1)
len(random_graph.edges())
is_connected(random_graph)

# Exercise: What do you think it means for a directed graph to be connected? Write a function that checks whether a directed graph is connected
#TODO:


# Probability of connectivity
# This function takes n and p, generates iters graphs, and returns the fraction of them that are connected.
# version with a for loop
def prob_connected(n, p, iters=100):
    count = 0
    for i in range(iters):
        random_graph = make_random_graph(n, p)
        if is_connected(random_graph):
            count += 1
    return count/iters

# version with a list comprehension
def prob_connected(n, p, iters=100):
    tf = [is_connected(make_random_graph(n, p))
            for i in range(iters)]
    return np.mean(tf)
# With n=10 and p=0.23, the probability of being connected is about 33%
np.random.seed(17)
n = 10
prob_connected(n, 0.23, iters=10000)

# the critical value of p for n=10 is about 0.23
pstar = np.log(n)/n
pstar

# So let's plot the probability of connectivity for a range of values for p
ps = np.logspace(-1.3, 0, 11)
ps

# estimate the probability with iters=1000
ys = [prob_connected(n,p,1000) for p in ps]
for p,y in zip(ps, ys):
    print(p, y)
# then plot them, adding a vertical line at the computed critical value
plt.axvline(pstar, color='gray')
plt.plot(ps, ys, color='green')
decorate(xlabel='Prob of edge (p)',
        ylabel='Prob connected',
        xscale='log')
savefig('figs/chap02-5')

# run the same analysis for a few more values of n
ns = [300, 100, 30]
ps = np.logspace(-2.5, 0, 11)

sns.set_palette('Blues_r', 4)
for n in ns:
    print(n)
    pstar = np.log(n) / n
    plt.axvline(pstar, color='gray', alpha=0.3)

    ys = [prob_connected(n, p) for p in ps]
    plt.plot(ps, ys, label='n=%d' % n)

decorate(xlabel='Prob of edge (p)',
         ylabel='Prob connected',
         xscale='log',
         xlim=[ps[0], ps[-1]],
         loc='upper left')
savefig('figs/chap02-6')

# Exercises








