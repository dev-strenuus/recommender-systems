from acoc import acoc
from kmeans import kmeans

class Main:

    graph = None

    @staticmethod
    def run():
        filepath = 'input.txt'  
        with open(filepath) as fp:  
            objects = []
            line = fp.readline()
            while line:
                xy = line.split(" ")
                objects.append((int(xy[0]), int(xy[1])))
                line = fp.readline()
        
        solver = acoc.ACOCsolver()
        solver.solve(objects)
    
Main.run()