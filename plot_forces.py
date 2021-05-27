import os
import matplotlib.pyplot as plt
import pandas


DIR = r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\forces"

def main():
    files = [f for f in os.listdir(DIR)
             if f.endswith(".csv")]
    lengths = []
    means = []
    variances = []
    for f in files:
        df = pandas.read_csv(os.path.join(DIR, f))
        length = os.path.basename(f).strip(".csv")
        if "_" in length:
            length = length.split("_")[0]
        lengths.append(int(length))
        means.append(df.Results.mean())
        variances.append(df.Results.var())

    plt.errorbar(lengths, means, variances, fmt=".")
    plt.show()


if __name__ == "__main__":
    main()