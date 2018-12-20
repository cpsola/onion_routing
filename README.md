# Setup

    pip install -r requirements.txt
    
# Running:

    usage: m_paper_exp.py [-h] [-f DATA_FOLDER] [-p PICKLE_FOLDER] [-l PATH_LEN]
                      [-c COST] [-i ITS] [-g GRAPH_NAME] [-t STARTING_TIME]

    
### Paper's experiments:

If data is in the `data` folder, results are stored in `pickle_files` folder, `cost` is 1, 
`path_len` is 5, and we are using the 100 selected nodes as source nodes in the experiments:
 
At zero hour:

    time python m_paper_exp.py -g Seattle32.txt -t 0

    
At rush hour:

    time python m_paper_exp.py -g Seattle32.txt -t 1


