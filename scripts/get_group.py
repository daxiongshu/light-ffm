def get_group(name):
    out = name.replace("lffm","ffm")+'.group'
    with open(name) as f:
	with open(out,"w") as fo:
	    count = 0
	    for c,line in enumerate(f):
		if line[0]=='x':
		    count += 1
		else:		   
		    fo.write("%d,%s\n"%(count,line[0]))	

if __name__ == "__main__":
    get_group("tr.lffm")
    get_group("va.lffm")
