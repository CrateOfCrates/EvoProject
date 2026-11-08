import numpy as np
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="get embedding comps")

    parser.add_argument("comp1", type=str, help="path to file 1")
    parser.add_argument("comp2", type=str, help="path to file 2")
    parser.add_argument("--bounds", type=int, default=25, help="boundaries for visulatizaton")

    args = parser.parse_args()


    cut_site1 = args.comp1.translate(str.maketrans({'.': ' ', '_': ' '})).split(" ")[2]
    cut_site2 = args.comp2.translate(str.maketrans({'.': ' ', '_': ' '})).split(" ")[2]
    try:
        if(int(cut_site1) == int(cut_site2)):
            print("These are matching extracted embeddings!")
        else:
            print("These embeddings do not match in cut site")
    except:
        raise Exception("Filenames are incorrect!")


    matrix_to_plot1 = np.load(args.comp1)
    matrix_to_plot2 = np.load(args.comp2)
    
    diff_matrix = matrix_to_plot1-matrix_to_plot2

    #diff_matrix = diff_matrix[int(cut_site1)-args.bounds:int(cut_site1)+args.bounds, :]

    # matrix_min = diff_matrix.min()
    # matrix_max = diff_matrix.max()
    # diff_matrix = (diff_matrix - matrix_min) / (matrix_max - matrix_min)

    plt.figure(figsize=(15, 9))
    plt.imshow(diff_matrix, aspect='auto', cmap='twilight')
    plt.ylim(int(cut_site1)-args.bounds, int(cut_site1)+args.bounds)
    plt.xlim(0, 2000)
    plt.show()

    
if __name__ == "__main__":
    main()