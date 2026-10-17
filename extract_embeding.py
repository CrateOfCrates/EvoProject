import argparse
import torch
import matplotlib.pyplot as plt;
import numpy as np
from evo2 import Evo2
from Bio.Seq import Seq
import os


def get_embeddings(seq1, evo2_model, device, layer):

    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(str(seq1)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    outputs, embeddings = evo2_model(
        input_ids, 
        return_embeddings=True, 
        layer_names=[layer]
    )

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

    #check for any problems that may occur before running
    if(len(args.replacement) != 1 or not (args.replacement) in "TAGC"):
        raise Exception("Replacement nucleotide is invalid, either incorrect length or char")

    if(args.cut_site < 0):
        raise Exception("Cut site is out of bounds, too small")

    if(args.cut_site > len(sequence1)-1):
            raise Exception("Cut site is out of bounds, too big")

    if(sequence1[args.cut_site] == args.replacement):
        raise Exception("Replacement nucletoide is the same as one to be replaced")

    # Verify CUDA visibility inside WSL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Evo2-1b
    try: 
        evo2_model = Evo2('evo2_1b_base', local_path='model/evo2_1b_base/evo2_1b_base.pt')
    except:
        raise Exception("Evo2 model loading process failed. Check local path.")

    #create the second sequence, replaces the cut site spot with the value
    sequence2 = sequence1[:args.cut_site] + args.replacement + sequence1[args.cut_site + 1 :]

    #create antisense sequence counterparts
    antisense_seq = Seq(sequence1);
    sequence1_opposite = antisense_seq.reverse_complement()
    antisense_seq = Seq(sequence2);
    sequence2_opposite = antisense_seq.reverse_complement()

    # choose the Layer to examine
    layer_name = 'blocks.'+(str)(args.target_layer)+'.mlp'

    # extract embeddings
    matrix1 = get_embeddings(sequence1, evo2_model, device, layer_name)
    matrix2 = get_embeddings(sequence2, evo2_model, device, layer_name)
    matrix1_opp = get_embeddings(sequence1_opposite, evo2_model, device, layer_name)
    matrix2_opp = get_embeddings(sequence2_opposite, evo2_model, device, layer_name)

    difference_matrix = matrix1-matrix2;
    difference_matrix_opp = matrix1_opp-matrix2_opp;

    # get variables
    total_size = difference_matrix.shape[0] - 1
    print("Spot to cut: " + (str)(args.cut_site))

    # flip sense matrix diff
    difference_matrix_opp = np.flip(difference_matrix_opp, axis=0)

    #cut down to size
    difference_matrix_opp = difference_matrix_opp[0:args.cut_site, :]
    difference_matrix = difference_matrix[args.cut_site:total_size, :]

    #combine
    combined_diff_matrix = np.vstack((difference_matrix_opp, difference_matrix))

    #print
    print("Combined Difference Matrix:", combined_diff_matrix.shape)

    try:
        target = os.path.join(str(args.file_path_dest).strip().rstrip(','), "na.npy");
        np.save(target, combined_diff_matrix)
        print("Finishing save")
    except:
        print("Invalid or nonexistant desitiantion, not saving anything")
    finally:
        print("Complete")

if __name__ == "__main__":
    main()

    