#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <boost/python.hpp>
#include "crc32c.h"


using namespace std;
using namespace boost::python;
using namespace tf_crc32;

unsigned calc_mask_crc(object o)
{
	string src = boost::python::extract<string>(o);
	unsigned mask_crc = MaskedCrc(src.c_str(), src.size());
	return mask_crc;
}


BOOST_PYTHON_MODULE(tfcrc)
{
	using namespace boost::python;
	def("calc_mask_crc", calc_mask_crc);
}

