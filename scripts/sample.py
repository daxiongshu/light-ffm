from random import random

def print_help():
    print "Usage: python sample.py [input file] [ratio] \nFor example: python sample.py tr.ffm 0.2 #random sample 20% samples and write into tr_0.2.ffm \n python sample.py va.lffm 0.5 # random sample 20% samples and write into va_0.2.lffm \n"

def sample_ffm(inputx, group, ratio):
    last = inputx.split('.')[-1]
    output = inputx.replace(".%s"%last,"_%.3f.%s"%(ratio,last)) 
    outg = output+'.group'
    print output
    fg = open(group)
    fog = open(outg,"w")
    last = ''
    write = False
    with open(inputx) as f:
	with open(output,"w") as fo:
	    for c,line in enumerate(f):
		gline = fg.readline()
		gg = gline.split(',')[0]
		
		if last!=gg:
		    if random()<ratio:
		    	write = True
		if write:
		    fo.write(line)
		    fog.write(gline)
		last = gg
		if c>0 and c%100000 == 0:
		    print "%s %d lines processed"%(inputx, c)

def sample_lffm(inputx, ratio):
    last = inputx.split('.')[-1]
    output = inputx.replace(".%s"%last,"_%.3f.%s"%(ratio,last))
    print output
    with open(inputx) as f:
        with open(output,"w") as fo:
	    write=False
            for c,line in enumerate(f):
		if line[0] == 'x':
                    if random()<ratio:
			write = True
		    else:
			write = False
		if write:
                    fo.write(line)
		
                if c>0 and c%100000 == 0:
                    print "%s %d lines processed"%(inputx, c)

def sample(mode,inputx,ratio):
    if mode == "ffm":
	sample_ffm(inputx, ratio)
    elif mode == "lffm":
	sample_lffm(inputx, ratio)
    else:
	print_help()


if __name__ == "__main__":
    #print_help()
    sample_ffm("tr.ffm",0.1)
    sample_ffm("va.ffm",0.1)
