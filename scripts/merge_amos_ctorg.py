import numpy as np
import socket
import os
import json
import SimpleITK as sitk


if socket.gethostname() == 'DESKTOP-HROVR50':
    # on local computer
    raw_dir = "E://Guided Research/AMOS22/"
    processed_dir = 'E://Guided Research/AMOS22_preprocessed/'

else:
    # on polyaxon
    raw_dir = "/data/dan_blanaru/AMOS22/"
    processed_dir = "/data/dan_blanaru/AMOS22_preprocessed/"


# generate json for ctorg
# resize AMOS script accomodates CT_ORG
# combine them