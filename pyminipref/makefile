CXX ?= g++

r: libminisat.so
d: libminisat.so

r: CFLAGS=-std=c++11 -fpic -D NDEBUG -O3 -Wall -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -Wno-parentheses -Wextra
d: CFLAGS=-std=c++11 -fpic -D DEBUG -O0 -ggdb -Wall -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -Wno-parentheses -Wextra

OS=$(shell uname -s)
ifeq ($(OS), Darwin)
	SHARED=-dynamiclib
else
	SHARED=-shared
endif

SATINC=minisat/

libminisat.so: minisat.o satSolver.o satSystem.o
	$(CXX) $(SHARED) $(CFLAGS) -o $@ $^

minisat.o: minisat.cpp
	$(CXX) -c $(CFLAGS) -I $(SATINC) -o $@ $^

satSolver.o: minisat/minisat/core/Solver.cc
	$(CXX) -c $(CFLAGS) -I $(SATINC) -o $@ $^

satSystem.o: minisat/minisat/utils/System.cc
	$(CXX) -c $(CFLAGS) -I $(SATINC) -o $@ $^
    
clean:
	rm -f *.so *.o

# check for existence of python versions to control tests
PYTHON2 := $(shell command -v python 2> /dev/null)
PYTHON3 := $(shell command -v python3 2> /dev/null)

test:
ifdef PYTHON2
	python test_pyminipref.py
	python -m doctest -v pyminipref.py
endif
ifdef PYTHON3
	python3 test_pyminipref.py
	python3 -m doctest -v pyminipref.py
endif
	@echo
	@echo "[32mAll tests passed.[m"

.PHONY: clean test
