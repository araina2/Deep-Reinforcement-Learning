
import seaborn as sns
import matplotlib.pyplot as plt

def loadNumericSeriesFromFile(filePath):
    input_file = open(filePath, 'r')

    # fp = open(filename, "r")
    # lines = fp.read().split("\n")
    lines = input_file.readlines()
    lines = filter(None, lines)
    lines = lines[19:]

    x = []
    y = []
    numPoints = 0

    for line in lines:
        if line.startswith('episode'):
            tokens = line.split('\t')
            x.append(float(tokens[1]))
            y.append(float(tokens[5]))
            numPoints += 1

    return y, x


def getMovingAverages(mylist, runLength):
	avgs = []
	for i in xrange(0,len(mylist)-runLength-1):
		my_sum=sum(mylist[i:i+runLength-1])
		my_avg=my_sum/runLength
		avgs.append(my_avg)
	return avgs


if __name__ == "__main__":

    xs = []
    ys = []
    legends = []

	# Score 
    filename = 'conv_lstm.txt'
    y, x = loadNumericSeriesFromFile('./'+filename)
    # y.sort()

    ys.append(y)
    xs.append(x)
    legends.append('Score')

	# Moving Average  -100
    runLength=100
    my_avgs=getMovingAverages(y, runLength)
    ys.append(my_avgs)
    xs.append(x[0:len(x)-runLength-1])
    legends.append('Moving Avg - ' + str(runLength))
    


    sns.set(style="darkgrid")
    for i in range(0,len(ys)):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)


    plt.ylim(ymin=0)
    # plt.legend(legends, fontsize=15, loc='best')
    plt.legend(legends, fontsize=15, loc='upper right')
    plt.title('Alien - CNN with LSTM', fontsize=20)
    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Score', fontsize=15)


    plt.autoscale(enable=True)
    plt.show()