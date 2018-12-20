import simple_paths_onion as spo
from constants import SEP

import networkx as nx
import numpy as np
import pickle
import sys
import json

from scipy.special import entr
from random import choice, sample, randint
from sys import exit
from os import listdir
from os.path import isfile, join


def get_graph(filename, data_folder):
    """
    Loads an edge file into a network directed multigraph.

    :param filename: name of the graph file
    :param data_folder: folder where the graph file is found
    :return: networkx graph
    """
    g = nx.MultiGraph()
    with open(data_folder + "/" + filename) as fp:
        line = fp.readline()
        while line:
            (o, d, t, e) = line.split()
            g.add_edge(int(o), int(d), start=int(t), duration=int(e))
            line = fp.readline()
    return g


def describe_graph(g):
    """
    Prints basic info about a networkx graph g.
    :param g: networkx graph
    :return:
    """
    print("Order: {} nodes".format(g.number_of_nodes()))
    print("Max node id: {}".format(max([n for n in g.nodes()])))
    print("Size: {} edges (interactions)".format(g.number_of_edges()))
    print("Density: {}".format(nx.density(g)))
    ts = nx.get_edge_attributes(g, 'start')
    ds = nx.get_edge_attributes(g, 'duration')
    print("First timestamp is: {}".format(min(ts.values())))
    print("Last timestamp is: {}".format(max([ts[k] + ds[k] for k in ts.keys()])))


def compute_anon_set_size(m):
    """
    Computes the size of the anonimity set.

    :param m: m: number of paths matrix, matrix[i][j] has the number of times a path from i arrives to j
    :return: array, anonimity set size of each source node
    """
    return (m > 0).sum(axis=1, keepdims=True)


def compute_entropy(m, nan_as_zero=True):
    """
    Given a num_of_paths_matrix, compute the entropy per rows.

    :param m: number of paths matrix, matrix[i][j] has the number of times a path from i arrives to j
    :param nan_as_zero: if a row of 0s is found, assume entropy is 0 (instead of NaN)
    :return: array of entropies, each value is the entropy of a row
    """
    # normalize rows (the sum of each row must be 1)
    p = m / m.sum(axis=1, keepdims=True)
    # compute per row entropy (in base 2)
    e = entr(p).sum(axis=1) / np.log(2)
    if nan_as_zero:
        e = np.nan_to_num(e)
    return e


def compute_anon_degree(m):
    """
    Computes anonymity degree given the entropy array.

    :param e: entropy array, as returned by compute_entropy
    :return: anonimity degree array
    """
    e = compute_entropy(m)
    return e / np.log2(m.shape[1])


def compute_all_paths_of_len(g, sampl, start_times, path_len=5, cost=1):

    raise DeprecationWarning
    exit()

    # TODO: use sparse matrices or just store the data we need
    # ids may not be sequential, so just create a bigger matrix (we delete the unused rows afterwards)
    n = max(g.nodes()) + 1
    num_of_paths_matrix = np.zeros((n, n), dtype=float)

    for source, start_time in zip(sampl, start_times):
        print "Processing source node: {} (starting at time {})".format(source, start_time)
        for dest in g.nodes():
            paths = [p for p in spo.all_simple_paths(g, source, dest,
                                                     cutoff=path_len, start_time=start_time, cost=cost)]
            for p in paths:
                assert(len(p) == path_len + 1)
            num_of_paths_matrix[source, dest] = len(paths)

    # delete unused rows and columns (non sequential ids)
    to_delete = [x for x in range(n) if x not in g.nodes()]
    num_of_paths_matrix = np.delete(num_of_paths_matrix, to_delete, axis=0)
    num_of_paths_matrix = np.delete(num_of_paths_matrix, to_delete, axis=1)

    return num_of_paths_matrix


def get_subsample_of_nodes(g, sampl=1):
    """
    Randomly selects a fraction `sampl` of nodes of the graph g
    :param g: networkx graph
    :param sampl: fraction of nodes to select
    :return: list with the selected nodes
    """
    return sample(g.nodes(), int(len(g.nodes())*sampl))


def get_random_timestamps(s, when="zero"):
    """
    Randomly selects starting timestamps for a list of source nodes s.

    :param s: list of nodes
    :param when: "zero" or "rush" (interval of valid values)
    :return: list with randomly selected timestamps
    """
    interval = [0, 10000] if when == "zero" else [25000, 35000]
    return [randint(*interval) for _ in s]


def compute_m_paths_of_len(g, sampl, start_times, path_len=5, cost=1, its=100, max_duration=sys.maxint):
    """
    Computes the number of paths of length `path_len` that start at each node (row) and end at each node (column).

    For each source node, `its` number of paths are randomly chosen, and the count to each destination
        node is stored in the resulting matrix. In the resulting matrix, matrix[i][j] stores de number
        of times a path starting in i ends up in j.

    If sampl=1, all nodes are used as source nodes in the path search. Otherwise, `sampl` indicates the fraction
        of nodes to use as starting nodes.

    The sum of each row should be equal to `its`, but may be less if invalid paths have been found.

    :param g: networkx graph (directed multigraph with edges having attributes "start" and "duration")
    :param sampl: list, subsample of nodes of the graph that will be used as starting nodes
    :param start_times: list, starting times for each source node
    :param path_len: int, length of the path
    :param cost: int, cost of transversing an edge
    :param its: int, number of iterations to try
    :param max_duration: int, max duration of the path
    :return: matrix, matrix[i][j] stores de number of times a path starting in i ends up in j.
    """

    num_of_paths_dict = {}

    # for each node in source, generate random paths and count the number of times they end up in each
    # dest. node
    for source, start_time in zip(sampl, start_times):
        print "Processing source node: {} (starting at time {})".format(source, start_time)
        num_of_paths_dict[(source, start_time)] = {dest: 0 for dest in g.nodes()}
        for _ in range(its):
            d = random_path_v1(g, source, path_len, start_time, cost, max_duration)
            if d is not None:
                num_of_paths_dict[(source, start_time)][d] += 1

    return num_of_paths_dict


def dict_to_np_matrix(d):

    row_order = sorted(d.keys())
    col_order = sorted(d[row_order[0]].keys())
    num_of_paths_matrix = np.zeros((len(row_order), len(col_order)), dtype=float)
    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            num_of_paths_matrix[i, j] = d[r][c]

    return num_of_paths_matrix, row_order, col_order


def random_path_v1(g, source, path_len, start_time=0, cost=1, max_duration=sys.maxint):
    """
    Selects a random path (i.e. without repeating nodes) of length `path_len` starting at
        node `source`.

    :param g: networkx graph (directed multigraph with edges having attributes "start" and "duration")
    :param source: int, node
    :param path_len: int, path length
    :param start_time: int, starting time of the path
    :param cost: int, cost of transversing an edge
    :return: int, destination node or None (if the random path turned to be invalid)
    """

    # TODO: maybe we can do a v2 that tries to generate another path if the current path is not valid
    # (although we have to take into account that such path may not exist)

    current_time = start_time
    path = [source]
    for _ in range(path_len):
        e = list(g.edges(path[-1], data=True))
        if len(e):
            n = choice(e)
            if n[1] in path:
                #  print "Node already visited"
                return None
            if n[2]["start"] + n[2]["duration"] >= current_time + cost:
                current_time = max(current_time + cost, n[2]["start"] + cost)
                if current_time - start_time < max_duration:
                    path.append(n[1])
                else:
                    # print "Too long path"
                    return None
            else:
                # print "Invalid time path"
                return None
        else:
            # print "No neighbors"
            return None

    # print path
    return path[-1]


def print_current_params(args):

    print(SEP)
    print("Parameters:")
    print("\tgraph:\t\t\t{}".format(args.graph_name))
    print("\tpath_len:\t\t{}".format(args.path_len))
    print("\tcost:\t\t\t{}".format(args.cost))
    print("\titerative\t\t{}".format(args.its))
    print("\ttime\t\t\t{}".format("rush" if args.starting_time else "zero"))
    print(SEP)


def load_guille_json_to_matrix(data_folder, graph_name, time, max_node_id=1179):
    """

    :param data_folder:
    :param graph_name:
    :param time:
    :return:
    """

    def get_graph_name(filename):
        return filename.split("_")[2]

    def get_graph_time(filename):
        return filename.split("_")[3]

    def get_source_node(filename):
        return int(filename.split("_")[8])

    def is_prob(filename):
        return "PROB" in filename

    files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and
             is_prob(f) and get_graph_name(f) == graph_name and get_graph_time(f) == time]

    nodes = [get_source_node(f) for f in files]
    n = max_node_id + 1
    num_of_paths_matrix = np.zeros((n, n), dtype=float)
    destinations = set()
    for f in files:
        with open(join(data_folder, f)) as of:
            data = json.load(of)
            source = get_source_node(f)
            for k, v in data.items():
                num_of_paths_matrix[source, int(k)] = float(v)
                destinations.add(int(k))

    # TODO: change representation! we can not store data in a matrix like this if sources nodes are repeated
    # delete unused rows (non sequential ids + not chosen in subsample)
    num_of_paths_matrix = num_of_paths_matrix[nodes, :]
    print(num_of_paths_matrix.shape)

    # delete unused columns (non sequential ids)
    to_delete = [x for x in range(n) if x not in destinations]
    num_of_paths_matrix = np.delete(num_of_paths_matrix, to_delete, axis=1)

    print(num_of_paths_matrix.shape)
    return num_of_paths_matrix


def load_pickle_file(data_folder, graph_name, its=1000000, path_len=5, time=0):
    """

    :param data_folder:
    :param graph_name:
    :param its:
    :param path_len:
    :param time:
    :return:
    """

    filename = join(data_folder, "{}_{}_{}_{}_1.0.pickle".format(graph_name, path_len, its, time))
    p = pickle.load(open(filename, "rb"))
    return p["m_aprox"], p["a_aprox"], p["e_aprox"]


def compare_results():
    m_aprox, _, _ = load_pickle_file("pickle_files", "Seattle32", time=0)
    m = load_guille_json_to_matrix("paths_for_anonymity_degree_N6", "N6", "zero")

    print m.shape
    print m_aprox.shape
    exit()


