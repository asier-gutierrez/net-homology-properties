import itertools
from collections import Counter
import numpy as np
import copy
import networkx as nx
import igraph as ig
from tensorflow.keras.layers import Dense
from gtda.graphs import GraphGeodesicDistance


def model2graphig(model, method=None, layer_activation_link=False, visualize=False, include_bias=True,
                  min_edge_distance=0.000001):
    MIRROR_STR = '_m'
    layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    G = ig.Graph(directed=True)

    # get max and min weight
    max_weights_total = 0
    min_weights_total = 0
    for idx, layer in enumerate(layers):
        weights, weights_bias = layer.get_weights()  # tiene que haber una manera más elegante de hacer esto concatenando arrays de pesos
        if max(weights.flatten()) > max_weights_total:
            max_weights_total = max(weights.flatten())
        if min(weights.flatten()) < min_weights_total:
            min_weights_total = min(weights.flatten())
        # Bias
        if include_bias:
            if max(weights_bias.flatten()) > max_weights_total:
                max_weights_total = max(weights_bias.flatten())
            if min(weights_bias.flatten()) < min_weights_total:
                min_weights_total = min(weights_bias.flatten())
    max_abs_weight = max(abs(max_weights_total), abs(min_weights_total))

    def compose_bias_name(idx, x):
        return 'b_' + str(idx) + '_' + str(x)

    def compose_node_name(idx, x):
        return 'n_' + str(idx) + '_' + str(x)

    activation_counts = Counter(list(
        itertools.chain(*[layer.activation._keras_api_names * len(layer.get_weights()[0][1]) for layer in layers])))
    first = True
    last_layers = None
    for idx, layer in enumerate(layers):
        idx = idx + 1
        weights, weights_bias = layer.get_weights()
        previous_node_names = list(map(lambda x: compose_node_name(idx - 1, x), range(weights.shape[0])))
        node_names = list(map(lambda x: compose_node_name(idx, x), range(weights.shape[1])))
        prevnn, currnn = list(zip(*list(itertools.product(previous_node_names, node_names))))
        bias_names = list(map(lambda x: compose_bias_name(idx, x), range(len(weights_bias))))
        edges = list(zip(prevnn, currnn))

        def connect_layer_activation(G, node_names, activation_count):
            connections = list(itertools.permutations(node_names, 2))
            scores = [1 / len(node_names) + 1 / activation_count] * len(connections)
            G.add_edges(connections)
            G.es[-len(connections):]['weight'] = scores

        if not method:
            # Weights
            if first:
                G.add_vertices(previous_node_names)
                first = False
            G.add_vertices(node_names)
            G.add_edges(edges)
            G.es[-len(edges):]['weight'] = weights.flatten(order='C')
            if layer_activation_link:
                connect_layer_activation(G, node_names, activation_counts[layer.activation._keras_api_names[0]])
            # Bias
            edges = list(zip(bias_names, node_names))
            G.add_vertices(bias_names)
            G.add_edges(edges)
            G.es[-len(edges):]['weight'] = weights_bias
        elif method == 'mirror':
            # Weights
            w = weights.flatten(order='C')
            if first:
                G.add_vertices(previous_node_names)
                G.add_vertices([p + MIRROR_STR for p in previous_node_names])
                first = False
            G.add_vertices(node_names)
            G.add_vertices([c + MIRROR_STR for c in node_names])
            G.add_edges(
                [(e[0], e[1]) if w[idx] > 0 else (e[0] + MIRROR_STR, e[1] + MIRROR_STR) for idx, e in enumerate(edges)])
            G.es[-len(currnn):]['weight'] = np.abs(w)
            # Bias
            G.add_vertices(bias_names)
            G.add_vertices([b + MIRROR_STR for b in bias_names])
            edges = list(zip(bias_names, node_names))
            G.add_edges([(e[0], e[1]) if weights_bias[idx] > 0 else (e[0] + MIRROR_STR, e[1] + MIRROR_STR) for idx, e in
                         enumerate(edges)])
            G.es[-len(edges):]['weight'] = np.abs(weights_bias)
        elif method == 'reverse':
            # Weights
            w = weights.flatten(order='C')
            G.add_vertices(previous_node_names)
            G.add_vertices(node_names)
            G.add_edges([(e[0], e[1]) if w[idx] > 0 else (e[1], e[0]) for idx, e in enumerate(edges)])
            G.es[-len(currnn):]['weight'] = np.maximum(1 - np.abs(w) / max_abs_weight, min_edge_distance)
            if layer_activation_link:
                connect_layer_activation(G, node_names, activation_counts[layer.activation._keras_api_names[0]])
            # Bias
            G.add_vertices(bias_names)
            edges = list(zip(bias_names, node_names))
            G.add_edges([(e[0], e[1]) if weights_bias[idx] > 0 else (e[1], e[0]) for idx, e in enumerate(edges)])
            G.es[-len(edges):]['weight'] = np.maximum(1 - np.abs(weights_bias) / max_abs_weight, min_edge_distance)
        last_layers = node_names
    G['last_layers'] = last_layers
    if visualize:
        # Thought for extremely small networks.
        layout = G.layout_auto()
        ig.plot(G, "./output/graph.pdf", layout=layout, bbox=(9000, 9000))

    return G


def graphig2geodesic(G):
    adjacencyMatrixCSR = G.get_adjacency_sparse(attribute='weight')
    AdjacencyMatrixCompleted = GraphGeodesicDistance().fit_transform(
        [adjacencyMatrixCSR])  # directed=True, unweighted=False
    G = ig.Graph.Weighted_Adjacency(AdjacencyMatrixCompleted[0].astype(np.float32).tolist(), attr="weight",
                                    loops=False)
    return G


def confmat2graph(confmat):
    confmat = copy.deepcopy(confmat)
    confmat = confmat / np.sum(confmat)
    np.fill_diagonal(confmat, 0)
    G = ig.Graph.Adjacency((confmat > 0).tolist())
    G.es['weight'] = confmat[confmat.nonzero()]
    return G


def attach_confmat(G, confmat):
    confmat = copy.deepcopy(confmat)
    confmat = confmat / np.sum(confmat)
    edges = \
        np.array(list(itertools.product(G['last_layers'], G['last_layers']))).reshape(confmat.shape[0],
                                                                                      confmat.shape[1],
                                                                                      2)[confmat.nonzero()]
    G.add_edges(edges)
    G.es[-len(edges):]['weight'] = confmat[confmat.nonzero()]
    return G


''' networkx
def model2graph(model):
    """
    Deprecated
    """
    smoothing = 0.000001
    layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    G = nx.DiGraph()

    def compose_bias_name(idx, x):
        return 'b_'+str(idx)+'_'+str(x)

    def compose_node_name(idx, x):
        return 'n_'+str(idx)+'_'+str(x)

    for idx, layer in enumerate(layers):
        idx = idx + 1
        weights, weights_bias = layer.get_weights()
        previous_node_names = list(map(lambda x: compose_node_name(idx - 1, x), range(weights.shape[0])))
        node_names = list(map(lambda x: compose_node_name(idx, x), range(weights.shape[1])))
        prevnn, currnn = list(zip(*list(itertools.product(previous_node_names, node_names))))

        G.add_weighted_edges_from(list(zip(prevnn, currnn, weights.flatten(order='C'))))
        bias_names = list(map(lambda x: compose_bias_name(idx, x), range(len(weights_bias))))
        G.add_weighted_edges_from(list(zip(bias_names, node_names, weights_bias)))

    return G
'''
