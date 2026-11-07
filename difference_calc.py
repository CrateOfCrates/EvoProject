import numpy as np
import argparse

def linearCKA(Xname, Yname):
        X = np.load(Xname)
        Y = np.load(Yname)
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # Calculate Gram matrices
        XTX = X.T.dot(X)
        YTY = Y.T.dot(Y)
        YTX = Y.T.dot(X)
        
        # Compute CKA numerator and denominator
        numerator = (YTX ** 2).sum()
        denominator = np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
        
        return numerator / (denominator + 1e-12)

def raw_likelihood(Xname, Yname):
        X = np.load(Xname)
        Y = np.load(Yname)
        result = -0.5*np.sum((X - Y) ** 2)
    
    # Return the raw difference: Likelihood(A) - Likelihood(B)
        return result

def unknown(): return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="get embedding comps")

    parser.add_argument("comp1", type=str, help="path to file 1")
    parser.add_argument("comp2", type=str, help="path to file 2")
    parser.add_argument("method", type=str, help="method to compare with")
    #parser.add_argument("--destination", type=str, help="path to file 2")

    args = parser.parse_args()

    result = 0.0

    match args.method:
        case "linearCKA":
            result = linearCKA(args.comp1, args.comp2)
        case "raw_likelihood":
            result = raw_likelihood(args.comp1, args.comp2)
        case _:
            print("No method found.")

    print(result)

    
if __name__ == "__main__":
    main()