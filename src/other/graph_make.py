"""
折れ線グラフを作るためのうんちコード 多分使わない

"""
import datetime
import jpholiday
import json
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import umap.umap_ as umap
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm

def main():

    ac_gan=[
0.5529,
0.6,
0.797,
0.7196,
0.5241,
0.6721,
0.5689,
0.5197,
0.52409,
0.5789,
0.6279

    ]
    cgan=[
0.6666,
0.6714,
0.647,
0.5585

    ]
    gan=[
0.9101,
0.8071,
0.8579,
0.7073,
0.6612,
0.6721,
0.7083,
0.6118,
0.5783,
0.6381,
0.6104

    ]
    l_gan=[
0.7083,
0.6785,
0.6764,
0.7446

    ]
    distance=[
13.0024,
5.9460,
4.6972,
2.4514,
2.3426,
2.2911,
1.9012,
1.8730,
1.7057,
1.6463,
1.5132

    ]
    # color=[
    #     "grey",
    #     "black",
    #     "grey",
    #     "black",
    #     "grey",
    #     "grey",
    #     "black",
    #     "black"
    # ]
    
    p_ac_gan, = plt.plot(distance, ac_gan, color="royalblue",zorder=1,linewidth = 3.0)
    # p_c_gan, = plt.plot(distance, cgan, color="cyan",zorder=2,linewidth = 3.0)
    p_gan, = plt.plot(distance, gan, color="orange",zorder=3,linewidth = 3.0)
    # p_l_gan, = plt.plot(distance, l_gan, color="firebrick",zorder=4,linewidth = 3.0)
    for i, n in enumerate(distance):
        plt.scatter(n, ac_gan[i],
                    color="royalblue", zorder=5)
        # plt.scatter(n, cgan[i],
        #             color="black", zorder=6)
        plt.scatter(n,  gan[i],
                    color="orange", zorder=7)
        # plt.scatter(n, l_gan[i],
        #             color="black", zorder=8)
    plt.legend((p_ac_gan, p_gan), ("AC-GAN", "GAN"), loc=2)
    plt.xscale("log", basex=2)
    plt.xlabel("distance")
    plt.ylabel("accuracy")
    # plt.title('')
    plt.ylim([0.5,0.95])


    #グラフを表示する
    plt.grid()
    plt.savefig("output/experiments/graph")

if __name__ == '__main__':
    main()