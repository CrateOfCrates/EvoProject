import argparse
import torch
import matplotlib.pyplot as plt;
import numpy as np
from evo2 import Evo2
from Bio.Seq import Seq
import os
import time


def get_embeddings(seq1, evo2_model, device, layer):

    start_time = time.perf_counter()

    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(str(seq1)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    outputs, embeddings = evo2_model(
        input_ids, 
        return_embeddings=True, 
        layer_names=[layer]
    )
    end_time = time.perf_counter()
    print(f"Finishing extraction in {end_time-start_time} seconds. ")

    return embeddings[layer].squeeze(0).detach().cpu().to(torch.float32).numpy()

def main():

    #parser (take in inputs)
    parser = argparse.ArgumentParser(description="extract embeddings")

    parser.add_argument("file_path_seq1", type=str, help="The path to the wild type file")
    parser.add_argument("cut_site", type=int, help="site to preform the cut, starts at index 0")
    parser.add_argument("replacement", type=str, help="check what to replace it with")
    
    parser.add_argument("--destination", type=str, help="The path where the file should be saved")
    parser.add_argument("--target_layer", type=int, default=17, help="The layer to be extracted")

    args = parser.parse_args()

    #extract string from the text file
    with open(str(args.file_path_seq1).strip().rstrip(','), "r", encoding="utf-8") as file:
        sequence1 = file.read().replace("\n", "")

    #check for any problems that may occur before running, unless rest are blank

    if(len(sequence1) == 0):
        raise Exception(f"Input sequence is empty; the file path to the sequence is: {args.file_path_seq1}")
    
    if(len(args.replacement) != 1 or not (args.replacement) in "TAGC"):
        raise Exception("Replacement nucleotide is invalid, either incorrect length or char")

    if(args.cut_site < 0):
        raise Exception("Cut site is out of bounds, too small")

    if(args.cut_site > len(sequence1)-1):
        raise Exception(f"Cut site is out of bounds, too big: {args.cut_site} >= {len(sequence1)}")

    if(sequence1[args.cut_site] == args.replacement):
        raise Exception("Replacement nucletoide is the same as one to be replaced")

    # Verify CUDA visibility inside WSL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Evo2
    try: 
        evo2_model = Evo2('evo2_7b', local_path='model/evo2_7b_base/evo2_7b.pt')
    except:
        raise Exception("Evo2 model loading process failed. Check local path.")

    #start timer
    start_time = time.perf_counter()

    #modify the sequence if necessary
    sequence2 = sequence1[:args.cut_site] + args.replacement + sequence1[args.cut_site + 1 :]

    #create antisense sequence counterparts
    antisense_seq = Seq(sequence1)
    sequence1_opposite = antisense_seq.reverse_complement()
    antisense_seq = Seq(sequence2)
    sequence2_opposite = antisense_seq.reverse_complement()

    # choose the Layer to examine
    layer_name = 'blocks.'+(str)(args.target_layer)+'.mlp'

    # extract embeddings
    wildtype = get_embeddings(sequence1, evo2_model, device, layer_name)
    modified = get_embeddings(sequence2, evo2_model, device, layer_name)
    wildtype_opp = get_embeddings(sequence1_opposite, evo2_model, device, layer_name)
    modified_opp = get_embeddings(sequence2_opposite, evo2_model, device, layer_name)

    # get variables
    total_size = wildtype.shape[0] - 1
    print("Spot to cut for both: " + (str)(args.cut_site))

    # flip sense matrix diff
    wildtype_opp = np.flip(wildtype_opp, axis=0)
    modified_opp = np.flip(modified_opp, axis=0)

    #cut down to size
    wildtype_opp = wildtype_opp[0:args.cut_site, :]
    wildtype = wildtype[args.cut_site:total_size, :]
    modified_opp = modified_opp[0:args.cut_site, :]
    modified = modified[args.cut_site:total_size, :]

    #combine
    combined_wildtype = np.vstack((wildtype_opp, wildtype))
    combined_modify = np.vstack((modified_opp, modified))

    #delete the actuall cut site row, helps reduce effect of missense comp to silent
    combined_wildtype = np.delete(combined_wildtype, args.cut_site-1, axis=0)
    combined_modify = np.delete(combined_modify, args.cut_site-1, axis=0)

    #end timer
    end_time = time.perf_counter()

    targetWT = os.path.join(str(args.destination).strip().rstrip(','), f"L{args.target_layer}_wildtype_cut{args.cut_site}.npy")
    targetM = os.path.join(str(args.destination).strip().rstrip(','), f"L{args.target_layer}_cut{args.cut_site}_{sequence1[args.cut_site]}-to-{args.replacement}.npy")

    try:
        np.save(targetWT, combined_wildtype)
        np.save(targetM, combined_modify)
    except:
        print("Invalid or nonexistant desitiantion, not saving anything")
    finally:
        print(f"Process completed in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

    