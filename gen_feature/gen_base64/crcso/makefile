.PHONY: all
all: tfcrc.so test

crc32c.o: crc32c.cc
	g++ -c -O2 -fPIC crc32c.cc
test: crc32c.o
	g++ -O2 -o test test.cpp crc32c.o

tfcrc.so: crc32c.o tfcrc.cpp
	g++ -shared -o tfcrc.so -fPIC -O2 -I/home/jd_ad/bailu/soft/archieve/boost_1_60_0 -I/home/jd_ad/bailu/soft/python27/include/python2.7 -L/home/jd_ad/bailu/boost_lib -lboost_python tfcrc.cpp crc32c.o
	cp tfcrc.so ../

clean:
	/bin/rm -fr crc32c.o test tfcrc.so ../tfcrc.so
