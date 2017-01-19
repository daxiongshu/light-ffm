num = 18 
fo = open('feamap.txt','w')
for i in range(num):
    line = ' '.join([str(j) for j in range(num)])
    fo.write('%d: %s\n'%(i,line))
fo.close()
	
