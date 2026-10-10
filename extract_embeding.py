import argparse
import torch
import matplotlib.pyplot as plt;
import numpy as np
from evo2 import Evo2
from Bio.Seq import Seq
import os


def main():
    parser = argparse.ArgumentParser(description="extract embeddings")

    parser.add_argument("file_path_seq1", type=str, help="The path to the wild type file")
    parser.add_argument("file_path_seq2", type=str, help="The path to the mutant type file")
    parser.add_argument("file_path_dest", type=str, help="The path where the file should be saved")

    parser.add_argument("--target_layer", type=int, default=17, help="The layer to be extracted")

    args = parser.parse_args()

    # Verify CUDA visibility inside WSL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Evo2-1b
    evo2_model = Evo2('evo2_1b_base', local_path='model/evo2_1b_base/evo2_1b_base.pt')

    # with open("data/wildtype.txt", "r", encoding="utf-8") as file:
    #     sequence1 = file.read().replace("\n", "")
    # with open("data/benign_missense.txt", "r", encoding="utf-8") as file:
    #     sequence2 = file.read().replace("\n", "")

    with open(str(args.file_path_seq1).strip().rstrip(','), "r", encoding="utf-8") as file:
        sequence1 = file.read().replace("\n", "")
    with open(str(args.file_path_seq2).strip().rstrip(','), "r", encoding="utf-8") as file:
        sequence2 = file.read().replace("\n", "")

    #create sense sequence counterparts
    antisense_seq = Seq(sequence1);
    sequence1_opposite = antisense_seq.reverse_complement()
    antisense_seq = Seq(sequence2);
    sequence2_opposite = antisense_seq.reverse_complement()

    print(sequence1_opposite)
    print(sequence2_opposite)

    # Get embeddings from Evo2-1b

    # Layer to examine
    layer_name = 'blocks.'+(str)(args.target_layer)+'.mlp'

    input_ids1 = torch.tensor(
        evo2_model.tokenizer.tokenize(str(sequence1)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    input_ids2 = torch.tensor(
        evo2_model.tokenizer.tokenize(str(sequence2)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    input_ids1_opposite = torch.tensor(
        evo2_model.tokenizer.tokenize(str(sequence1_opposite)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    input_ids2_opposite = torch.tensor(
        evo2_model.tokenizer.tokenize(str(sequence2_opposite)),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    outputs, embeddings1 = evo2_model(
        input_ids1, 
        return_embeddings=True, 
        layer_names=[layer_name]
    )

    outputs, embeddings2 = evo2_model(
        input_ids2, 
        return_embeddings=True, 
        layer_names=[layer_name]
    )

    outputs, embeddings1_opposite = evo2_model(
        input_ids1_opposite, 
        return_embeddings=True, 
        layer_names=[layer_name]
    )

    outputs, embeddings2_opposite = evo2_model(
        input_ids2_opposite, 
        return_embeddings=True, 
        layer_names=[layer_name]
    )

    embedding_matrix1 = embeddings1[layer_name]
    embedding_matrix2 = embeddings2[layer_name]
    embedding_matrix1_opposite = embeddings1_opposite[layer_name]
    embedding_matrix2_opposite = embeddings2_opposite[layer_name]


    print("Embedding matrix shape 1:", embedding_matrix1.shape)
    print("Embedding matrix shape 2:", embedding_matrix2.shape)
    print("Embedding matrix opposite shape 1:", embedding_matrix1_opposite.shape)
    print("Embedding matrix opposite shape 2:", embedding_matrix2_opposite.shape)

    matrix1 = embedding_matrix1.squeeze(0).detach().cpu().to(torch.float32).numpy()
    matrix2 = embedding_matrix2.squeeze(0).detach().cpu().to(torch.float32).numpy()
    matrix1_opp = embedding_matrix1_opposite.squeeze(0).detach().cpu().to(torch.float32).numpy()
    matrix2_opp = embedding_matrix2_opposite.squeeze(0).detach().cpu().to(torch.float32).numpy()

    difference_matrix = matrix1-matrix2;
    difference_matrix_opp = matrix1_opp-matrix2_opp;

    # get variables
    total_size = difference_matrix.shape[0] - 1
    print(total_size)
    cut_site = next((i for i, (a, b) in enumerate(zip(sequence1, sequence2)) if a != b), min(len(sequence1), len(sequence2)))
    print(cut_site)

    # flip sense matrix diff
    difference_matrix_opp = np.flip(difference_matrix_opp, axis=0)

    #cut down to size
    difference_matrix_opp = difference_matrix_opp[0:cut_site, :]
    difference_matrix = difference_matrix[cut_site:total_size, :]

    #combine
    combined_diff_matrix = np.vstack((difference_matrix_opp, difference_matrix))

    #print
    print("Combined Difference Matrix:", combined_diff_matrix.shape)

    try:
        target = os.path.join(str(args.file_path_dest).strip().rstrip(','), "na.npy");
        np.save(target, combined_diff_matrix)
    except:
        print("Invalid desitiantion")
    finally:
        print("Finishing save")



if __name__ == "__main__":
    main()

    