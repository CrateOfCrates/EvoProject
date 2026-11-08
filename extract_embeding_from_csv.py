from math import nan
import os
import sys
import ctypes
# Automatically preload the virtualenv NCCL library before PyTorch initializes
def _preload_nccl():
    venv_nccl = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2")
    if os.path.exists(venv_nccl):
        ctypes.CDLL(venv_nccl, mode=ctypes.RTLD_GLOBAL)
_preload_nccl()


import argparse
import time
import torch
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from evo2 import Evo2
from get_pos import getpostp53

def main():

    parser = argparse.ArgumentParser(description="extract_from_csv")

    #default start for TP53 is 7687490, based on file
    #Wildtype is based on the COMPLIMENT strand, SPDI is based on the FORWARD strand
    parser.add_argument("file_name", type=str, help="CSV file name")
    parser.add_argument("location_name", type=str, help="name for csv column for ID")
    parser.add_argument("original_mark_name", type=str, help="name for csv column for original nucleotide")
    parser.add_argument("replacement_mark_name", type=str, help="name for csv column for replacemenent nucleotide")
    parser.add_argument("wild_type", type=str, help="wild type file name")
    parser.add_argument("desti", type=str, help="desitrnatrion")
    args = parser.parse_args()

    df = pd.read_csv(f"{args.file_name}")
    
    locations = df[args.location_name]
    original = df[args.original_mark_name]
    replace = df[args.replacement_mark_name]

    with open(str(args.wild_type).strip().rstrip(','), "r", encoding="utf-8") as file:
        sequence1 = file.read().replace("\n", "")

    # Verify CUDA visibility inside WSL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Evo2
    try: 
        evo2_model = Evo2('evo2_7b', local_path='model/evo2_7b_base/evo2_7b.pt')
    except:
        raise Exception("Evo2 model loading process failed. Check local path.")


    #PRIMARY LOOP
    snv_count = 0
    for i in range(len(locations)):
        
        if(not(len(str(original[i])) == 1 and len(str(replace[i])) == 1)):
            print("NOT A SNV")
            continue

        print(f"{getpostp53(locations[i])} where {original[i]} becomes {replace[i]}")
        snv_count += 1


    print(f"Finished processed {snv_count} SNV")


print("start")
if __name__ == "__main__":
    main()