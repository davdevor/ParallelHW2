#
# Bridges - PSC
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CC = g++
MPCC = mpicxx
OPENMP = -fopenmp
CFLAGS = -O3
LIBS =


TARGETS = serial openmp mpi 

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
openmp: openmp.o commonopenmp.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o commonopenmp.o
mpi: mpi.o commonmpi.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o commonmpi.o

openmp.o: openmp.cpp commonopenmp.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
mpi.o: mpi.cpp commonmpi.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp
commonmpi.o: commonmpi.cpp commonmpi.h
	$(CC) -c $(CFLAGS) commonmpi.cpp
commonopenmp.o: commonopenmp.cpp commonopenmp.h
	$(CC) -c $(OPENMP) $(CFLAGS) commonopenmp.cpp
clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
