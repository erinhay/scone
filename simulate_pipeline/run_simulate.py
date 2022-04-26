### IMPORTS ###
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
import os
import time
from new_simulate_utils import *
import argparse
import yaml

### CONFIG ###
parser = argparse.ArgumentParser(description='set up the SCONE model')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
parser.add_argument('--obj_typ', type=int, help='specify which class of objects to simulate with int key')
parser.add_argument('--label_start', type=int, help='specify integer to start labeling simulated objects by')
parser.add_argument('--num_objects_per_job', type=int, help='number of objects to simulate in this job')
parser.add_argument('--i', type=int, help='label')
args = parser.parse_args()

with open(args.config_path, "r") as cfgfile:
    config = yaml.safe_load(cfgfile)

METADATA_PATHS = config["metadata_paths"]
LCDATA_PATHS = config["lcdata_paths"]
SIM_PATH = config["simulate_path"]
NUM_OF_EACH_TYPE = args.num_objects_per_job #config["num_of_each_class"]
NUM_CLASSES = config["num_types"]
#model_lcs = dict(config["model_lcs"])
#model_lcs = {key: value for key, value in model_lcs.items() if value == args.obj_typ}

#print(model_lcs)
    
### MAIN ###
print('opening data')
#print(len(METADATA_PATHS))
for i, (metadata_path, lcdata_path) in enumerate(zip(METADATA_PATHS, LCDATA_PATHS)):
    single_meta = pd.read_csv(metadata_path)
    single_lcdata = pd.read_csv(lcdata_path)
    
    if i == 0:
        metadata = single_meta
        lcdata = single_lcdata
    else:
        metadata = pd.concat([single_meta, metadata])
        lcdata = pd.concat([single_lcdata, lcdata])
        
model_lcs = dict(zip(metadata['object_id'], metadata['true_target']))
model_lcs = {key: value for key, value in model_lcs.items() if value == args.obj_typ}

print(model_lcs)

#metadata = metadata[metadata['ddf_bool'] == 1]

### DATA ###
band0_err = lcdata[lcdata['passband'] == 0][['mjd','flux','flux_err']]
band1_err = lcdata[lcdata['passband'] == 1][['mjd','flux','flux_err']]
band2_err = lcdata[lcdata['passband'] == 2][['mjd','flux','flux_err']]
band3_err = lcdata[lcdata['passband'] == 3][['mjd','flux','flux_err']]
band4_err = lcdata[lcdata['passband'] == 4][['mjd','flux','flux_err']]
band5_err = lcdata[lcdata['passband'] == 5][['mjd','flux','flux_err']]

print('starting simulation')
start = time.time()
simulated_lcdata = Table(names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
simulated_metadata = Table(names=('model_id', 'true_target', 'object_id', 'sim_z', 'sim_timeshift'))

possible_ids = metadata[args.label_start:args.label_start+NUM_OF_EACH_TYPE]['object_id']
print(possible_ids[args.label_start])
simulated_ids = []

fail = 0
tot_sim = 0

while len(simulated_metadata) < NUM_OF_EACH_TYPE:
    model_id = np.random.choice(list(model_lcs.keys()))
    
    while metadata[metadata['object_id'] == model_id]['true_peakmjd'] - np.min(ldata[ldata['object_id'] == model_id]['mjd']) < 20:
        model_id = np.random.choice(list(model_lcs.keys()))
    
    #if len(simulated_metadata) > 1:
        #if len(simulated_metadata[simulated_metadata['true_target'] == model_lcs.get(model_id)]) >= NUM_OF_EACH_TYPE:
            #continue
    
    #print(len(simulated_metadata))
    #print(model_id)
    
    for ID in np.random.choice(possible_ids, 1):
        if ID in simulated_ids:
            continue
        
        tot_sim += 1
        
        simulated_sndata, simulated_snmeta, _ = simulate_event(lcdata, metadata, model_id, ID, band0_err, band1_err, band2_err, band3_err, band4_err, band5_err)
    
        if str(simulated_sndata) == 'None':
            fail += 1
            continue
        else:
            print(ID)
            simulated_lcdata = vstack([simulated_lcdata, simulated_sndata])
            simulated_metadata = vstack([simulated_metadata, simulated_snmeta])
            simulated_ids.append(ID)
            print(len(simulated_metadata))

print('writing outputs')
ascii.write(simulated_lcdata, "{}/plasticc{}_{}_simulated_lightcurve.csv".format(SIM_PATH, args.obj_typ, args.i), overwrite=True)
ascii.write(simulated_metadata, "{}/plasticc{}_{}_simulated_metadata.csv".format(SIM_PATH, args.obj_typ, args.i), overwrite=True)

end = time.time()

print('finished simulation')

print('time to simulate ' + str(tot_sim) + ' objects: ' + str(end-start) + ' sec')

print('done')