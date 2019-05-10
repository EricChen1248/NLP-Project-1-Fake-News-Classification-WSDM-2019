import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




def main():
    df = pd.read_csv("../data/train.csv")
    print(df.shape)
    label = []
    for i in range(df.shape[0]):
        if (df.loc[i, "label"] == "unrelated"):
            label.append(0)
        elif (df.loc[i, "label"] == "agreed"):
            label.append(1)
        else:
            label.append(2)
    print(np.array(label))
    label = np.array(label)
    np.save("../../label.npy", label)
    

if __name__ == '__main__':
	main()