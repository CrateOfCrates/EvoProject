import numpy as np
import pickle
import gzip
import argparse
import time
from tests.CKA_Centered_Kernel_Alignment.CKA import linear_CKA, kernel_CKA
from tests.codebook_examples.cka.cka_step3 import cka

def raw_likelihood(X, Y):
        result = -0.5*np.sum((X - Y) ** 2)
        return result

def unknown(): return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="get embedding comps")

    parser.add_argument("comp1", type=str, help="path to file 1")
    parser.add_argument("comp2", type=str, help="path to file 2")
    parser.add_argument("method", type=str, help="method to compare with")
    parser.add_argument("--bounds", type=int, default=50, help="boundaries for cutting down")
    #parser.add_argument("--destination", type=str, help="path to file 2")

    args = parser.parse_args()

    result = 0.0

    X = np.load(str(args.comp1))
    Y = np.load(str(args.comp2))
    print("Finished Loading files")

    #find location of SNV
    cut_site1 = args.comp1.translate(str.maketrans({'.': ' ', '_': ' '})).split(" ")[2]
    cut_site2 = args.comp2.translate(str.maketrans({'.': ' ', '_': ' '})).split(" ")[2]
    try:
        if(int(cut_site1) == int(cut_site2)):
            print("These are matching extracted embeddings!")
        else:
            print("These embeddings do not match in cut site")
    except:
        raise Exception("Filenames are incorrect!")

    #
    X = X[int(cut_site1)-args.bounds:int(cut_site1)+args.bounds+1, :]
    Y = Y[int(cut_site1)-args.bounds:int(cut_site1)+args.bounds+1, :]

    start_time = time.perf_counter()

    match args.method:
        case "linearCKA":
            result = linear_CKA(X, Y)
        case "cka":
            result = cka(X, Y)
        case "raw_likelihood":
            result = raw_likelihood(X, Y)
        case _:
            print("No method found.")

    end_time = time.perf_counter()

    print(f"{result} comparision using {args.method} in {end_time-start_time} seconds.")

    
if __name__ == "__main__":
    main()