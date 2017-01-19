CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: ffm-train-predict 

ffm-train-predict: ffm-train-predict.cpp ffm.o FileUtil.o StringUtil.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp 
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $^

StringUtil.o: StringUtil.cpp StringUtil.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

FileUtil.o: FileUtil.cpp FileUtil.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<


clean:
	rm -f  ffm-train-predict  *.o
