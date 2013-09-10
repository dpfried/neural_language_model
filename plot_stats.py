import pandas
import matplotlib.pyplot as plt

constant_stats = pandas.load('logistic/constant/stats.pkl')
sve_stats = pandas.load('logistic/sve/stats.pkl')
def plot_stats(stats):
    plt.subplot(211)
    stats['test_mean'].plot()
    plt.subplot(212)
    stats['embedding_sim_mean'].plot()
plot_stats(constant_stats)
plt.suptitle('constant')

plt.figure()
plot_stats(sve_stats)
plt.suptitle('sve')
plt.show()
