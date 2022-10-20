import os
import json
import socket

if socket.gethostname() == 'DESKTOP-HROVR50':
    #on local computer
    raw_dir = "E://Guided Research/AMOS22/"
    processed_dir = 'E://Guided Research/AMOS22_preprocessed/'
else: 
    # on polyaxon
    raw_dir = "/data/dan_blanaru/AMOS22/"
    processed_dir = "/data/dan_blanaru/AMOS22_preprocessed/"

json_filename = "task1_dataset.json"
json_toy_filename = "toy_dataset.json"
json_read_path = processed_dir+json_filename
json_write_path = processed_dir+json_toy_filename

json_obj = json.load(open(json_read_path,'r'))

json_obj['training'] = json_obj['training'][:10]
json_obj['numTraining'] = 10
json_obj['test'] = json_obj['test'][:2]
json_obj['numTest'] = 2

json.dump(json_obj,open(json_write_path,'w'))
