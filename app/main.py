import os
from glob import glob
from parser import parse_file

DATAPATH = os.getcwd() + '/data/fitfiles/'

# Demonstrate parsing of files into dataframes
dfs = []
for file_name in glob(DATAPATH + '*.fit'):
    df = parse_file(file_name)
    dfs.append(df)
