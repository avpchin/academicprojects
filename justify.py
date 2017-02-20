# Code that justifies text
# (c) 2017 Kar Yern Chin

#!/usr/bin/env python
import math
#A skeleton pretty-printing file that reads in an integer and a
#file and creates a list of all the words in the file. This program
# then prints each word to standard out.

#Make the system namespace available.
#We use the argv member of sys to get
#the command-line arguments
import sys

#Python uses whitespace for control. This throws a lot of people
#off at first, but over time, one finds it incredibly natural. Read this:
#1. <http://weblog.hotales.org/cgi-bin/weblog/nb.cgi/view/python/2005/02/19/1>
#2. <http://www.secnetix.de/~olli/Python/block_indentation.hawk>

def main():
    # grab the first argument and make it an integer
    # Note that argv[0] contains the program name
    L = int(sys.argv[1])
    print(L)
    print("You are requesting {} character display".format(L))

    # create a file handle to the input
    fin = open(sys.argv[2])

    # read() returns the input as one string.
    # split() is a method ofstring. By default it tokenizes the
    # input using *whitespace* as a delimeter
    # now words is a list of strings
    words = fin.read().split()

    print("The list of words is:\n{} ".format("\n".join(words)))
    justify(L, words)
    # read <http://effbot.org/pyfaq/tutor-what-is-if-name-main-for.htm>

# The method that justifies the string.
def justify(L, words):
    n = len(words)

    # extra space in a line after fitting words i to j
    lineCost = [[0 for x in range(n+1)] for x in range(n + 1)]
    # cost of each line (which is basically the square of values of values of lineCost)
    lineCost2 = [[0 for x in range(n + 1)] for x in range(n + 1)]
    # total cost for fitting words 1 to j in the line
    totalCost = [0 for x in range(n + 1)]
    # array used to keep track of word index separation that will be printed
    p = [0 for x in range(n + 1)]

    # this loop fills the lineCost array
    for i in range(1, n + 1):
        lineCost[i][i] = L - len(words[i - 1])
        for j in range(i + 1, n + 1):
            lineCost[i][j] = lineCost[i][j - 1] - len(words[j-1]) - 1

    # this loop fills the lineCost2 array
    for i in range(1,n + 1):
        for j in range(i, n + 1):
            if lineCost[i][j] < 0:
                # if the words i to j do not fit into a line (negative value), turns it into infinity so that it won't be used
                lineCost2[i][j] = float("inf")
            elif j == n and lineCost[i][j] >= 0:
                lineCost2[i][j] = 0
            else:
                lineCost2[i][j] = lineCost[i][j] * lineCost[i][j]
    totalCost[0] = 0
    
    # this loop fills the totalCost array
    for j in range(1, n + 1):
        totalCost[j] = float("inf")
        for i in range(1, j + 1):
            if (totalCost[i-1] != float("inf")) and (lineCost2[i][j] != float("inf")) and (totalCost[i-1] + lineCost2[i][j] < totalCost[j]):
                totalCost[j] = totalCost[i-1] + lineCost2[i][j]
                p[j] = i

    # calls print solution method
    printSolution(p, len(words), words)

# helper method that prints the lines
def printSolution(p, n, words):
    k = 0
    string = ""
    if (p[n] == 1):
        k = 1
    else:
        k = printSolution(p, p[n] - 1, words) + 1
    for i in range(p[n] - 1, n):
        if (i == n):
            string = string + words[i]
        else:
            string = string + words[i] + " "
    if (string != ""):
        print(string)
    return k

if __name__ == '__main__' :
    main()

