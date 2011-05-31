# The generated "Makefile" from "Makefile.cmake" is only usable after
# the Armadillo library has been configured and installed by CMake.

CXX=g++
#CXX=g++-4.2
## Under MacOS you may have an old compiler as default (e.g. GCC 4.0).
## However, GCC 4.2 or later is available and preferable due to better
## handling of template code.

#CXX=CC
## When using the Sun Studio compiler

# /----------------------------------------------------------
# | Configuration for Intel Math Kernel Library
# +----------------------------------------------------------
# | Ignore on first reading
# |
MKL_PATH = /share/apps/intel/Compiler/11.1/046/mkl/
MKL_LIB_PATH = ${MKL_PATH}/lib/em64t/

# cf. MKL "link line advisor":
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/

MKL_LINK_LINE = -L${MKL_LIB_PATH} \
  ${MKL_LIB_PATH}/libmkl_lapack.so \
  ${MKL_LIB_PATH}/libmkl_core.so \
  ${MKL_LIB_PATH}/libmkl_intel_lp64.so \
  -Wl,--start-group -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -fopenmp -lpthread
# |
# \----------------------------------------------------------

LIBSCL_PATH = /home/dge218/usr/include/libscl/gpp
LIBSCL_LINE = -L${LIBSCL_PATH} -lscl  

# flags configured by CMake
ifeq (unix,macos)
  EXTRA_LIB_FLAGS = -framework Accelerate
endif

#EXTRA_LIB_FLAGS = -library=sunperf
## When using the Sun Studio compiler


ifeq (false,true)
  BOOST_INCLUDE_FLAG = -I Boost_INCLUDE_DIR-NOTFOUND
endif



LIB_FLAGS =${LIBSCL_LINE} ${MKL_LINK_LINE} $(EXTRA_LIB_FLAGS)
## NOTE: on Ubuntu and Debian based systems you may need to add 
## -lgfortran to LIB_FLAGS



OPT = -O2 -fopenmp
## As the Armadillo library uses recursive templates,
## compilation times depend on the level of optimisation:
##
## -O0: quick compilation, but the resulting program will be slow
## -O1: produces programs which achieve most of the possible speedup
## -O3: produces programs which have almost all possible speedups,
##      but compilation takes considerably longer


#OPT = -xO4 -xannotate=no
## When using the Sun Studio compiler


#EXTRA_OPT = -fwhole-program
## Uncomment the above line if you're compiling 
## all source files into one program in a single hit.


#DEBUG = -DARMA_EXTRA_DEBUG
## Uncomment the above line to enable low-level
## debugging.  Lots of debugging information will
## be printed when a compiled program is run.
## Please enable this option when reporting bugs.


#FINAL = -DARMA_NO_DEBUG
## Uncomment the above line to disable Armadillo's checks.
## DANGEROUS!  Not recommended unless your code has been
## thoroughly tested.


#
#
#

CXXFLAGS =-ggdb -I/home/dge218/usr/include/libscl/gpp -I/home/dge218/usr/include/ $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: GLExtension

GLExtension: main.o Economy.o Firm.o linearpoly.o
	$(CXX) $(CXXFLAGS) -o $@ $< Economy.o Firm.o linearpoly.o $(LIB_FLAGS)

main.o: main.cpp Economy.h
	$(CXX) $(CXXFLAGS) -c main.cpp

Economy.o: Economy.h Firm.h Economy.cpp
	$(CXX) $(CXXFLAGS) -c Economy.cpp

Firm.o: Firm.cpp Firm.h linearpoly.h linearinterpolator.h
	$(CXX) $(CXXFLAGS) -c Firm.cpp

linearpoly.o: linearpoly.h linearpoly.cpp
	$(CXX) $(CXXFLAGS) -c linearpoly.cpp


.PHONY: clean

clean:
	rm -f Firm.o Economy.o main.o linearpoly.o GLExtesion

