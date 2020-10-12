import random

with open("train_flist.txt") as fp:
    lines = fp.readlines()

random.shuffle(lines)
with open("train_flist.txt", "w") as fp:
    fp.writelines(lines)