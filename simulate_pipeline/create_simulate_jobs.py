import os
import yaml
import argparse
import subprocess
import numpy as np
import datetime
import pandas as pd

SBATCH_HEADER = """#!/bin/bash
#SBATCH -C haswell
#SBATCH --qos=shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=48:00:00
#SBATCH --output={log_path}

conda activate plasticcEnv
cd {output_path}
python /global/cscratch1/sd/erinhay/plasticc/scone/run_simulate.py --config_path {config_path} --obj_typ {obj_typ} --label_start {label_start} --num_objects_per_job {num_objects_per_job} --i {i}"""

parser = argparse.ArgumentParser(description='create simulated data from lightcurve templates')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_simulate_config.yml"')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.safe_load(cfgfile)
    return config

config = load_config(args.config_path)

LOG_OUTPUT_PATH = config['simulate_path']
METADATA_PATHS = config["metadata_paths"]
num_types = config["num_types"]
obj_typ_dict = dict(config["sn_type_id_to_name"])
obj_types = np.array(list(obj_typ_dict.keys()))

os.chdir(LOG_OUTPUT_PATH)

print('opening data')
#print(len(METADATA_PATHS))
for i, metadata_path in enumerate(METADATA_PATHS):
    single_meta = pd.read_csv(metadata_path)
    
    if i == 0:
        metadata = single_meta
    else:
        metadata = pd.concat([single_meta, metadata])

num_jobs_per_class = 4
num_of_each_class = int(len(metadata)/num_types)
label_starts = np.linspace(0, len(metadata)-1, len(obj_types)*num_jobs_per_class)
obj_types = np.tile(obj_types, num_jobs_per_class)
num_objects_per_job = int(label_starts[1]-label_starts[0])

num_simultaneous_jobs = 16 # haswell has 16 physical cores
print("num types: {}".format(num_types))
print("num of each type: {}".format(num_of_each_class))
print(obj_types)

for i, obj_typ in enumerate(obj_types):
    
    label_start = int(label_starts[i])
    
    SBATCH_FILE = os.path.join(LOG_OUTPUT_PATH, "autogen_simulate_batchfile_{obj_typ}_{i}.sh")
    
    sbatch_setup_dict = {
        "output_path": LOG_OUTPUT_PATH,
        "config_path": args.config_path,
        "log_path": f"CREATE_SIMULATE_{obj_typ}_{i}.log",
        "obj_typ": obj_typ,
        "label_start": label_start,
        "num_objects_per_job": num_objects_per_job,
        "i": i
    }
    sbatch_setup = SBATCH_HEADER.format(**sbatch_setup_dict)
    sbatch_file = SBATCH_FILE.format(**{"obj_typ": obj_typ, "i":i})
    with open(sbatch_file, "w+") as f:
        f.write(sbatch_setup)
    print(f"launching job {obj_typ} {i}")
    subprocess.run(["sbatch", sbatch_file])

