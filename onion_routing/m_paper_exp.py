from anonymity_set_netwx import *
from constants import *

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='data_folder', default="data")
    parser.add_argument('-p', dest='pickle_folder', default="pickle_files_new")
    parser.add_argument('-l', dest='path_len', default=4, type=int)
    parser.add_argument('-c', dest='cost', default=1, type=int)
    parser.add_argument('-i', dest='its', default=1000000, type=int)
    parser.add_argument('-g', dest='graph_name', default="Seattle32.txt")
    parser.add_argument('-t', dest='scenario_time', default=0, type=int)
    parser.add_argument('-r', dest='reverse', action='store_true')

    try:
        args = parser.parse_args()
        print_current_params(args)
    except:
        parser.print_help()
        exit(0)

    g = get_graph(args.graph_name, args.data_folder)
    # describe_graph(g)

    str_scenario_time = "rush" if args.scenario_time else "zero"
    if args.reverse:
        # backward path: we start from destination
        s = ENDING_NODES[str_scenario_time]
        st_origin = STARTING_TIMES[str_scenario_time]
        md = MAX_DURATION_TIME[str_scenario_time][NETWORKS_INV[args.graph_name[:-4]]]
        st = [t + md for t in st_origin]
        d_aprox = compute_m_paths_of_len_back(g, s, st, path_len=args.path_len, cost=args.cost, its=args.its,
                                              max_duration=md)
    else:
        # forward path: we start from origin
        s = STARTING_NODES[str_scenario_time]
        st = STARTING_TIMES[str_scenario_time]
        md = MAX_DURATION_TIME[str_scenario_time][NETWORKS_INV[args.graph_name[:-4]]]
        d_aprox = compute_m_paths_of_len_forw(g, s, st, path_len=args.path_len, cost=args.cost, its=args.its,
                                              max_duration=md)

    m_aprox, row_indx, col_indx = dict_to_np_matrix(d_aprox)
    assert m_aprox.shape[0] == 100
    assert m_aprox.shape[1] == len(g.nodes())
    e_aprox = compute_entropy(m_aprox)
    a_aprox = compute_anon_degree(m_aprox)
    as_aprox = compute_anon_set_size(m_aprox)

    print(SEP)
    print "Path matrix:\n{}".format(m_aprox)
    # print "Entropy:\n{}".format(e_aprox)
    print("Mean entropy: {}".format(np.mean(e_aprox)))
    # print "Anon. degree:\n{}".format(a_aprox)
    print("Mean anon. degree: {}".format(np.mean(a_aprox)))
    # print "Anon. set size:\n{}".format(as_aprox)
    print("Mean anon. set size: {}".format(np.mean(as_aprox)))

    pickle_name = "{}/{}_{}_{}_{}_{}.pickle".format(
        args.pickle_folder, args.graph_name[:-4], str_scenario_time, args.path_len, args.its, args.cost, )

    pickle.dump({"d_aprox": d_aprox, "m_aprox": m_aprox, "e_aprox": e_aprox, "a_aprox": a_aprox},
                open(pickle_name, "wb"))