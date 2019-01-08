from anonymity_set_netwx import *
from constants import *

if __name__ == "__main__":

    g = get_graph("Seattle32.txt", "data")

    dg = load_guille_csv("paths_for_anonymity_degree_N6", "result_anonimitydegreeTree_N6_zero_20181217131647.csv")
    m_aprox, row_indx, col_indx = guille_dict_to_np_matrix(dg, g)

    assert m_aprox.shape[0] == 100
    e_aprox = compute_entropy(m_aprox)
    a_aprox = compute_anon_degree(m_aprox)
    as_aprox = compute_anon_set_size(m_aprox)

    print("Mean entropy: {}".format(np.mean(e_aprox)))
    print("Mean anon. degree: {}".format(np.mean(a_aprox)))
    print("Mean anon. set size: {}".format(np.mean(as_aprox)))

    dc, m_aprox, a_aprox, e_aprox = load_pickle_file("pickle_files", "Seattle32", "zero", 5, 1000000, 1)
    as_aprox = compute_anon_set_size(m_aprox)
    print("----")
    print("Mean entropy: {}".format(np.mean(e_aprox)))
    print("Mean anon. degree: {}".format(np.mean(a_aprox)))
    print("Mean anon. set size: {}".format(np.mean(as_aprox)))

    for source_st, destinations in dc.items():
        for destination, num_times in destinations.items():
            if num_times != 0:
                if destination not in dg[source_st] or dg[source_st][destination] == 0:
                    print("I have {} path(s) between {} and {} (starting time {}) and guille doesn't".format(
                        num_times, source_st[0], destination, source_st[1]))
                    assert(False)

    i = 0
    while True:
        i += 1
        d = random_path_forward(g, 780, 5, 8623, 1, 65985)
        print i
        if d == 3:
            break

    # m = load_guille_probs_to_matrix("paths_for_anonymity_degree_N6", "N6", "zero")
