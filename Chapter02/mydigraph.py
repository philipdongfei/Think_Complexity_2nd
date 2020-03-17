import networkx as nx
import matplotlib.pyplot as plt
from complete_graph import all_pairs, make_complete_graph
COLORS=['red','green','blue','black']
G = nx.DiGraph()

G.add_node('Alice')
G.add_node('Bob')
G.add_node('Chuck')
G.add_edge('Alice', 'Bob')
G.add_edge('Alice', 'Chuck')
G.add_edge('Bob', 'Alice')
G.add_edge('Bob', 'Chuck')
nodes=list(G.nodes())
edges=list(G.edges())
#G.add_node(1)
#G.add_nodes_from([2,3])
'''

nodes = range(10)
G.add_nodes_from(nodes)
G.add_edges_from(all_pairs(nodes))
G = make_complete_graph(10)
'''

nx.draw_circular(G,
        node_color=COLORS[0],
        node_size=2000,
        with_labels=True)
plt.show()

