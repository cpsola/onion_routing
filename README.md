# Setup

    pip install -r requirements.txt
    
# Running:

    usage: anonymity_set_netwx.py [-h] [-g GRAPH_NAME] [-f DATA_FOLDER]
                              [-p PICKLE_FOLDER] [-l PATH_LEN] [-c COST]
                              [-d MAX_DURATION] [-e] [-i ITS]
                              [-r RANDM_SAMPLE] [-s SELECTED_SAMPLES]
    
    
### Paper's experiments:

If data is in the `data` folder, results are stored in `pickle_files` folder, `cost` is 1, 
`path_len` is 5, and we are using the 100 selected nodes as source nodes in the experiments:
 
At zero hour:

    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 65985 -g "Seattle32.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 68291 -g "Seattle16.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 69432 -g "Seattle08.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 70907 -g "Seattle04.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 70755 -g "Seattle02.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 0 -d 71436 -g "Seattle01.txt"
    
At rush hour:

    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 32070 -g "Seattle32.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 40915 -g "Seattle16.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 42473 -g "Seattle08.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 45310 -g "Seattle04.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 44421 -g "Seattle02.txt"
    time python anonymity_set_netwx.py -i 1000000 -s 1 -d 46442 -g "Seattle01.txt"

