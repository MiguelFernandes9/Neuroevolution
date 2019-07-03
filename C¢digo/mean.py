import glob, os


if __name__ == "__main__":

    global_mean = 0
    num_file = 0

    for file in glob.glob("*.txt"):
        rf = open(file,"r")
    
        stats = rf.read()

        stats = stats.split()

        sum = 0
        for i in stats:
            sum += float(i)
        mean = sum/len(stats)

        global_mean += mean

        num_file += 1

        print(mean, "file :" , file)

    print("Overall mean: ", global_mean/num_file)