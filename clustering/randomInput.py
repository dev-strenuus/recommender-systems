import random
f = open("input/input.txt", "w")
for i in range(0, 5000):
    f.write(str(random.randint(-10000,10000))+" "+str(random.randint(-10000,10000))+"\n")