#include <iostream>
#include <fstream>
#include "crc32c.h"

using namespace std;

int main()
{
	char header[sizeof(uint64) + sizeof(uint32)];
	unsigned len = 298;
	EncodeFixed64(header + 0, len);
	EncodeFixed32(header + sizeof(uint64),
			MaskedCrc(header, sizeof(uint64)));
	
	ofstream ofst("d.bin", std::ios::binary);
	ofst.write(header, 8 + 4);
	ofst.close();
	
	return 0;
}
