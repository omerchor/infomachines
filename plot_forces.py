import os
import matplotlib.pyplot as plt
import pandas
import numpy as np


DIR = r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\forces"


def compare_measurement_halfs():
    files = [f for f in os.listdir(DIR)
             if f.endswith(".csv")]
    for f in files:
        df = pandas.read_csv(os.path.join(DIR, f))
        results = df.Results
        times = df.Time
        a, b = np.polyfit(times, results, 1)
        print(f"F(t)={a:.2e}t+{b:.2f}")
        print(df.Results[:len(df.Results) // 2].max())
        print(df.Results[len(df.Results) // 2:].max())

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

    a, b = np.polyfit(lengths, means, 1)
    print(f"F(x)={a:.2e}x+{b:.2f}")
    xs = np.linspace(min(lengths), max(lengths), 100)

    plt.errorbar(lengths, means, variances, fmt=".")
    plt.plot(xs, a*xs + b)
    plt.xlabel("Arena Length (cm)")
    plt.ylabel("Average Force (N)")
    plt.show()


if __name__ == "__main__":
    main()
    # compare_measurement_halfs()


