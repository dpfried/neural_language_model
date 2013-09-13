import pandas
import matplotlib.pyplot as plt

def plot_stats(stats):
    plt.subplot(211)
    stats['test_mean'].plot()
    plt.subplot(212)
    stats['embedding_sim_mean'].plot()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_files', nargs='*')
    args = parser.parse_args()

    for stats_file in args.stats_files:
        stats = pandas.read_pickle(stats_file)
        plt.figure()
        plot_stats(stats)
        plt.suptitle(stats_file)
    plt.show()
