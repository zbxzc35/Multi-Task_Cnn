/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LIB_HASH_CRC32C_H_
#define TENSORFLOW_LIB_HASH_CRC32C_H_

#include <stddef.h>
#include <string.h>
#include <stdint.h>
//#include "tensorflow/core/platform/types.h"

namespace tf_crc32 {
const bool _kLittleEndian = true;
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

}

using namespace tf_crc32;

inline void EncodeFixed32(char* buf, uint32 value) {            
  //if (port::kLittleEndian) {                             
	if (_kLittleEndian) {
    memcpy(buf, &value, sizeof(value));                  
  } else {                                               
    buf[0] = value & 0xff;                               
    buf[1] = (value >> 8) & 0xff;                        
    buf[2] = (value >> 16) & 0xff;                       
    buf[3] = (value >> 24) & 0xff;                       
  }                                                      
}                                                        
                                                         
inline void EncodeFixed64(char* buf, uint64 value) {            
  //if (port::kLittleEndian) {                             
	if (_kLittleEndian) {
    memcpy(buf, &value, sizeof(value));                  
  } else {                                               
    buf[0] = value & 0xff;                               
    buf[1] = (value >> 8) & 0xff;                        
    buf[2] = (value >> 16) & 0xff;                       
    buf[3] = (value >> 24) & 0xff;                       
    buf[4] = (value >> 32) & 0xff;                       
    buf[5] = (value >> 40) & 0xff;                       
    buf[6] = (value >> 48) & 0xff;                       
    buf[7] = (value >> 56) & 0xff;                       
  }                                                      
}                                                        
                                                         

inline uint32 DecodeFixed32(const char* ptr) {
  //if (port::kLittleEndian) {
	if (_kLittleEndian) {
    // Load the raw bytes
    uint32 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64 DecodeFixed64(const char* ptr) {
  //if (port::kLittleEndian) {
  if (_kLittleEndian) {
    // Load the raw bytes
    uint64 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64 lo = DecodeFixed32(ptr);
    uint64 hi = DecodeFixed32(ptr + 4); 
    return (hi << 32) | lo; 
  }
}


//namespace tensorflow {
//namespace crc32c {

// Return the crc32c of concat(A, data[0,n-1]) where init_crc is the
// crc32c of some string A.  Extend() is often used to maintain the
// crc32c of a stream of data.
extern uint32 Extend(uint32 init_crc, const char* data, size_t n);

// Return the crc32c of data[0,n-1]
inline uint32 Value(const char* data, size_t n) { return Extend(0, data, n); }

static const uint32 kMaskDelta = 0xa282ead8ul;

// Return a masked representation of crc.
//
// Motivation: it is problematic to compute the CRC of a string that
// contains embedded CRCs.  Therefore we recommend that CRCs stored
// somewhere (e.g., in files) should be masked before being stored.
inline uint32 Mask(uint32 crc) {
  // Rotate right by 15 bits and add a constant.
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

// Return the crc whose masked representation is masked_crc.
inline uint32 Unmask(uint32 masked_crc) {
  uint32 rot = masked_crc - kMaskDelta;
  return ((rot >> 17) | (rot << 15));
}

inline uint32 MaskedCrc(const char* data, size_t n) {
  return Mask(Value(data, n));
}


//}  // namespace crc32c
//}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_HASH_CRC32C_H_
