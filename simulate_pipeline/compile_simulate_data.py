import pandas as pd
import os
import argparse
import yaml

print('initiating')
parser = argparse.ArgumentParser(description='set up the SCONE model')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
args = parser.parse_args()

with open(args.config_path, "r") as cfgfile:
    config = yaml.safe_load(cfgfile)
    
SIM_PATH = config["simulate_path"]
DATA_PATH = config["data_path"]

lc_path = os.path.join(DATA_PATH, 'simulated_lcdata.csv')
meta_path = os.path.join(DATA_PATH, 'simulated_metadata.csv')

simdata_folder = os.listdir(SIM_PATH)

print('compiling data file:')
simulated_lcdata = pd.DataFrame()
simulated_metadata = pd.DataFrame()

for entry in simdata_folder:
    if entry.endswith('lightcurve.csv'):
        print(entry)
        temp_lc = pd.read_csv(os.path.join(config["simulate_path"], entry), delimiter=' ')
        simulated_lcdata = pd.concat([simulated_lcdata, temp_lc])
    elif entry.endswith('metadata.csv'):
        print(entry)
        temp_meta = pd.read_csv(os.path.join(config["simulate_path"], entry), delimiter=' ')
        simulated_metadata = pd.concat([simulated_metadata, temp_meta])
    else:
        continue

print('saving compiled simulated data files')
simulated_lcdata.to_csv(lc_path)
simulated_metadata.to_csv(meta_path)

print('done')