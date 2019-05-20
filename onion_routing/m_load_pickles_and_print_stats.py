from anonymity_set_netwx import *
#from constants import *
import pickle
import os
import numpy as np

if __name__ == "__main__":

    pickle_folder = "pickle_files_back" # "pickle_files_forwd"

    for filename in os.listdir(pickle_folder):
        if filename.endswith(".pickle") and "_4_" in filename:
            try:
                print("------------------------------------")
                print(filename)
                data = pickle.load(open(pickle_folder + "/" + filename, "rb"))
                print("Mean entropy: {}".format(np.mean(data["e_aprox"])))
                print("Std. entropy: {}".format(np.std(data["e_aprox"])))
                print("Mean anon. degree: {}".format(np.mean(data["a_aprox"])))
                print("Std. anon. degree: {}".format(np.std(data["a_aprox"])))

                as_aprox = compute_anon_set_size(data["m_aprox"])
                print("Mean anon. set size: {}".format(np.mean(as_aprox)))
                print("Std. anon. set size: {}".format(np.std(as_aprox)))
                print("\n")
            except:
                print("Error\n")

