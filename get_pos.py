import numpy as np
import matplotlib.pyplot as plt
import argparse

def getpos(start, end): return start-end-1
def getpostp53(end): return 7687490-end-1

def main():

    parser = argparse.ArgumentParser(description="get positions")

    #default start for TP53 is 7687490, based on file
    #Wildtype is based on the COMPLIMENT strand, SPDI is based on the FORWARD strand
    parser.add_argument("--start", type=int, default=7687490, help="starting position")
    parser.add_argument("target", type=int, default=0, help="target position")
    parser.add_argument("file_path_seq1", type=str, help="The path to the wild type file")

    args = parser.parse_args()

    with open(str(args.file_path_seq1).strip().rstrip(','), "r", encoding="utf-8") as file:
        sequence1 = file.read().replace("\n", "")

    print(f"Target is: {getpos(args.start, args.target)}, which is base {sequence1[args.start - args.target - 1]}")

print("start")
if __name__ == "__main__":
    main()