import networkx as nx
import matplotlib.pyplot as plt
COLORS=['red','green','blue','black']

def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i > j:
                yield u, v
def make_complete_graph(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G

def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen
def is_connected(G):
    start = next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)


if __name__=='__main__':
    complete = make_complete_graph(10)
    nx.draw_circular(complete,
                 node_color=COLORS[0],
                 node_size=2000,
                 with_labels=True)
    plt.show()

    neighbors = complete.neighbors(0)
    for n in neighbors:
        print(n, end=' ')
    print()
    s = reachable_nodes(complete, 0)
    print(s)
    print(is_connected(complete))

