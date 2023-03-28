CXX = g++
# CXXFLAGS = -std=c++11 -O3 -march=native
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
# CXXFLAGS = -std=c++11 -O3 -mcpu=apple-m1

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)