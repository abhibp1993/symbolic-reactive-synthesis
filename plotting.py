import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("benchmark.pkl", "rb") as file:
        data = pickle.load(file)
    print(data)

    DIM = [d for d in range(5, 10)]
    plt.figure(0)

