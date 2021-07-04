infile = open("viterbi.txt","r")
outfile = open("vi.txt","w")
for eachline in infile.readlines():
    lines = filter(lambda ch: ch not in ' \t\r\n',eachline)
    outfile.write(lines)
outfile.close
infile.close

