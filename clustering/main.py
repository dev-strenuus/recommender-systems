from acoc import acoc
from kmeans import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler


class Main:

    graph = None

    @staticmethod
    def plot(clusters):
        x = []
        y = []
        c = []
        for i in range(0, len(clusters)):
            for j in range(0, len(clusters[i])):
                x.append(clusters[i][j][0])
                y.append(clusters[i][j][1])
                c.append(i)
        x = np.asarray(x)
        y = np.asarray(y)
        c = np.asarray(c)
        df = pd.DataFrame(dict(x=x, y=y, c=c))
        clusters = df.groupby('c')

        fig, ax = plt.subplots()
        for name, group in clusters:
            ax.plot(group.x, group.y, marker = 'o', linestyle='', ms=12, label=name)
        ax.legend()
        plt.show()

    @staticmethod
    def run():
        filepath = 'input/input.txt'  
        with open(filepath) as fp:  
            objects = []
            line = fp.readline()
            while line:
                xy = line.split(" ")
                objects.append((int(xy[0]), int(xy[1])))
                line = fp.readline()
        
        lp = LineProfiler()
        solver = acoc.ACOCsolver()
        clusters = solver.solve(objects, 4)
        Main.plot(clusters)

Main.run()