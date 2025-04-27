//
// Created by Kai Zhao on 1/28/21.
//

#ifndef SZ3_BYTEUTIL_HPP
#define SZ3_BYTEUTIL_HPP

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "SZ3/def.hpp"
// #include <xmmintrin.h>
// #include <immintrin.h>
#include <stddef.h>

namespace SZ3 {

typedef union lint16 {
    unsigned short usvalue;
    short svalue;
    unsigned char byte[2];
} lint16;

typedef union lint32 {
    int ivalue;
    unsigned int uivalue;
    unsigned char byte[4];
} lint32;

typedef union lint64 {
    int64_t lvalue;
    uint64_t ulvalue;
    unsigned char byte[8];
} lint64;

typedef union ldouble {
    double value;
    uint64_t lvalue;
    unsigned char byte[8];
} ldouble;

typedef union lfloat {
    float value;
    unsigned int ivalue;
    unsigned char byte[4];
    uint16_t int16[2];
} lfloat;

inline void symTransform_4bytes(uchar data[4]) {
    unsigned char tmp = data[0];
    data[0] = data[3];
    data[3] = tmp;

    tmp = data[1];
    data[1] = data[2];
    data[2] = tmp;
}

inline int16_t bytesToInt16_bigEndian(const unsigned char *bytes) {
    int16_t temp = 0;
    int16_t res = 0;

    temp = bytes[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[1] & 0xff;
    res |= temp;

    return res;
}

inline uint16_t bytesToUInt16_bigEndian(const uchar *bytes) {
    uint16_t temp = 0;
    uint16_t res = 0;

    temp = bytes[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[1] & 0xff;
    res |= temp;

    return res;
}

inline int32_t bytesToInt32_bigEndian(const unsigned char *bytes) {
    int32_t temp = 0;
    int32_t res = 0;

    res <<= 8;
    temp = bytes[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[1] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[2] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[3] & 0xff;
    res |= temp;

    return res;
}

inline uint32_t bytesToUInt32_bigEndian(const uchar *bytes) {
    uint32_t temp = 0;
    uint32_t res = 0;

    res <<= 8;
    temp = bytes[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[1] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[2] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[3] & 0xff;
    res |= temp;

    return res;
}

inline int64_t bytesToInt64_bigEndian(const unsigned char *b) {
    int64_t temp = 0;
    int64_t res = 0;

    res <<= 8;
    temp = b[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[1] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[2] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[3] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[4] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[5] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[6] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[7] & 0xff;
    res |= temp;

    return res;
}

inline uint64_t bytesToUInt64_bigEndian(const uchar *b) {
    uint64_t temp = 0;
    uint64_t res = 0;

    res <<= 8;
    temp = b[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[1] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[2] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[3] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[4] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[5] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[6] & 0xff;
    res |= temp;

    res <<= 8;
    temp = b[7] & 0xff;
    res |= temp;

    return res;
}

inline void int16ToBytes_bigEndian(unsigned char *b, int16_t num) {
    b[0] = (unsigned char)(num >> 8);
    b[1] = (unsigned char)(num);
}

inline void int32ToBytes_bigEndian(unsigned char *b, int32_t num) {
    b[0] = (unsigned char)(num >> 24);
    b[1] = (unsigned char)(num >> 16);
    b[2] = (unsigned char)(num >> 8);
    b[3] = (unsigned char)(num);
}

inline void int64ToBytes_bigEndian(unsigned char *b, int64_t num) {
    b[0] = (unsigned char)(num >> 56);
    b[1] = (unsigned char)(num >> 48);
    b[2] = (unsigned char)(num >> 40);
    b[3] = (unsigned char)(num >> 32);
    b[4] = (unsigned char)(num >> 24);
    b[5] = (unsigned char)(num >> 16);
    b[6] = (unsigned char)(num >> 8);
    b[7] = (unsigned char)(num);
}

std::string floatToBinary(float f) {
    lfloat u;
    u.value = f;
    std::string str(32, '0');
    for (int i = 0; i < 32; i++) {
        str[31 - i] = (u.ivalue % 2) ? '1' : '0';
        u.ivalue >>= 1;
    }
    return str;
}

template <class T>
void truncateArray(T data, size_t n, int byteLen, uchar *&binary) {
    lfloat bytes;
    int b;
    for (size_t i = 0; i < n; i++) {
        bytes.value = data[i];
        for (b = 4 - byteLen; b < 4; b++) {
            *binary++ = bytes.byte[b];
        }
    }
}

template <class T>
void truncateArrayRecover(uchar *binary, size_t n, int byteLen, T *data) {
    lfloat bytes;
    bytes.ivalue = 0;
    int b;
    for (size_t i = 0; i < n; i++) {
        for (b = 4 - byteLen; b < 4; b++) {
            bytes.byte[b] = *binary++;
        }
        data[i] = bytes.value;
    }
}

template <typename T>
uint8_t vector_bit_width(const std::vector<T> &data) {
    if (data.empty()) return 0;
    T max_value = *std::max_element(data.begin(), data.end());
    uint8_t bits = 0;
    while (max_value > 0) {
        max_value >>= 1;
        ++bits;
    }
    return bits;
}

template <typename T>
void vector2bytes(const std::vector<T> &data, uint8_t bit_width, unsigned char *&c) {
    if (data.empty()) return;

    size_t current_bit = 0;
    size_t byte_index = 0;
    unsigned char current_byte = 0;

    for (T value : data) {
        size_t bits_remaining = bit_width;
        while (bits_remaining > 0) {
            size_t space_in_current_byte = 8 - (current_bit % 8);
            size_t bits_to_write = std::min(bits_remaining, space_in_current_byte);
            size_t bits_shift = (bit_width - bits_remaining);
            unsigned char bits_to_store = (value >> bits_shift) & ((1 << bits_to_write) - 1);

            current_byte |= (bits_to_store << (current_bit % 8));
            current_bit += bits_to_write;
            bits_remaining -= bits_to_write;

            if (current_bit % 8 == 0) {
                c[byte_index++] = current_byte;
                current_byte = 0;
            }
        }
    }

    if (current_bit % 8 != 0) {
        c[byte_index++] = current_byte;
    }

    c += byte_index;
}

template <typename T>
std::vector<T> bytes2vector(const unsigned char *&c, uint8_t bit_width, size_t num_elements) {
    // uint8_t bit_width = *c++;

    std::vector<T> data(num_elements);

    size_t total_bits = num_elements * bit_width;
    size_t total_bytes = (total_bits + 7) / 8;

    for (size_t i = 0; i < num_elements; ++i) {
        T value = 0;
        for (uint8_t j = 0; j < bit_width; ++j) {
            size_t bit_index = i * bit_width + j;
            size_t byte_index = bit_index / 8;
            size_t bit_offset = bit_index % 8;

            value |= ((c[byte_index] >> bit_offset) & 1) << j;
        }
        data[i] = value;
    }

    c += total_bytes;

    return data;
}

// inline uchar* bitTranspose8(aligned_vector<int32_t> &in)
uchar* bitTranspose8(int32_t* in, size_t& in_size)
{
    if (in_size % 8 != 0) {
        int res = 8 - in_size % 8;
        for(int i = 0; i < res; i++) {in[in_size++] = 0; }
    }

    const size_t blockSize = 8;     
    const size_t bitsPerInt = 32;   
    size_t nBlocks = in_size / blockSize;

    uchar* out = static_cast<uchar*>(::operator new(nBlocks * bitsPerInt, std::align_val_t(512)));
    // uchar* out_B = static_cast<uchar*>(::operator new(nBlocks * bitsPerInt, std::align_val_t(256)));
    // #pragma omp parallel for
    for(size_t bit = 0; bit  < bitsPerInt; bit++) {
        // uint32_t mask = 1 << bit;
        for(size_t b = 0; b < nBlocks; b++) {
            size_t baseIn = b * blockSize;
            out[bit * nBlocks + b] = 
                            (((in[baseIn + 0] >> bit) & 1) << 7) |
                            (((in[baseIn + 1] >> bit) & 1) << 6) |
                            (((in[baseIn + 2] >> bit) & 1) << 5) |
                            (((in[baseIn + 3] >> bit) & 1) << 4) |
                            (((in[baseIn + 4] >> bit) & 1) << 3) |
                            (((in[baseIn + 5] >> bit) & 1) << 2) |
                            (((in[baseIn + 6] >> bit) & 1) << 1) |
                            (((in[baseIn + 7] >> bit) & 1));
        }
    }

    return out;
}

uchar* bitTranspose8inverse(aligned_vector<int32_t> &in)
{
    if (in.size() % 8 != 0) {
        int res = 8 - in.size() % 8;
        for(int i = 0; i < res; i++) {in.push_back(0); }
    }

    const size_t blockSize = 8;     
    const size_t bitsPerInt = 32;   
    size_t nBlocks = in.size() / blockSize;

    uchar* out = static_cast<uchar*>(::operator new(nBlocks * bitsPerInt, std::align_val_t(256)));
    // #pragma omp parallel for
    for(size_t bit = 0; bit  < bitsPerInt; bit++) {
        uint32_t mask = 1 << (bitsPerInt - bit - 1);
        for(size_t b = 0; b < nBlocks; b++) {
            size_t baseIn = b * blockSize;
            uint32_t in_0 = (in[baseIn + 0] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_1 = (in[baseIn + 1] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_2 = (in[baseIn + 2] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_3 = (in[baseIn + 3] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_4 = (in[baseIn + 4] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_5 = (in[baseIn + 5] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_6 = (in[baseIn + 6] & mask) >> (bitsPerInt - bit - 1);
            uint32_t in_7 = (in[baseIn + 7] & mask) >> (bitsPerInt - bit - 1);
            out[bit * nBlocks + b] = (in_0 << 7) | (in_1 << 6) | (in_2 << 5) | (in_3 << 4)
                | (in_4 << 3) | (in_5 << 2) | (in_6 << 1) | ((in_7 & 1u));
        }
    }

    return out;
}


inline void add_to_quant(aligned_vector<int32_t>& quant_inds, const aligned_vector<uchar>& bytes, int bitshift) {
    size_t intLen = quant_inds.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    int mod8 = intLen % 8;
    // #pragma omp parallel for
    for (size_t b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++) {
        size_t i = b * 8;
        uint32_t temp = bytes[b];
        uint32_t masks[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
        for (size_t j = 0; j < 8; ++j) {
            quant_inds[i + j] |= ((temp & masks[j]) >> (7 - j)) << bitshift;
        }
    }

    size_t i = intLen / 8 * 8, b = (mod8 == 0 ? byteLen : byteLen - 1);
    if (mod8 > 0) {
        if (mod8 >= 1) {
            quant_inds[i] += ((uint32_t) ((bytes[b] & 0x80) >> 7)) << bitshift;
        }
        if (mod8 >= 2) {
            quant_inds[i + 1] += ((uint32_t) ((bytes[b] & 0x40) >> 6)) << bitshift;
        }
        if (mod8 >= 3) {
            quant_inds[i + 2] += ((uint32_t) ((bytes[b] & 0x20) >> 5)) << bitshift;
        }
        if (mod8 >= 4) {
            quant_inds[i + 3] += ((uint32_t) ((bytes[b] & 0x10) >> 4)) << bitshift;
        }
        if (mod8 >= 5) {
            quant_inds[i + 4] += ((uint32_t) ((bytes[b] & 0x08) >> 3)) << bitshift;
        }
        if (mod8 >= 6) {
            quant_inds[i + 5] += ((uint32_t) ((bytes[b] & 0x04) >> 2)) << bitshift;
        }
        if (mod8 >= 7) {
            quant_inds[i + 6] += ((uint32_t) ((bytes[b] & 0x01))) << bitshift;
        }

    }
}

void add_to_quant(int32_t* quant_inds, size_t intLen, uchar* loaded_bits, int b_start, int b_end) {
    // size_t intLen = quant_inds.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    int mod8 = intLen % 8;
    // #pragma omp parallel for
    for (size_t b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++) {
        size_t i = b * 8;
        uint32_t adder[8] = {0};

        for(int bit = b_start; bit < b_end; bit++) {
            uchar loaded = loaded_bits[bit * byteLen + b];
            uint32_t mask = 1U << (31 - bit);

            // 使用位操作一次性处理所有 8 个 bits
            adder[0] |= (loaded & (0x80)) ? mask : 0;
            adder[1] |= (loaded & (0x40)) ? mask : 0;
            adder[2] |= (loaded & (0x20)) ? mask : 0;
            adder[3] |= (loaded & (0x10)) ? mask : 0;
            adder[4] |= (loaded & (0x08)) ? mask : 0;
            adder[5] |= (loaded & (0x04)) ? mask : 0;
            adder[6] |= (loaded & (0x02)) ? mask : 0;
            adder[7] |= (loaded & (0x01)) ? mask : 0;
        }

        // for(int ii = 0; ii < 8; ii++) {
        //     quant_inds[i + ii] |= adder[ii];
        // }

        quant_inds[i] += (adder[0] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 1] += (adder[1] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 2] += (adder[2] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 3] += (adder[3] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 4] += (adder[4] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 5] += (adder[5] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 6] += (adder[6] ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        quant_inds[i + 7] += (adder[7] ^ 0xaaaaaaaau) - 0xaaaaaaaau;


    }

    size_t i = intLen / 8 * 8, b = (mod8 == 0 ? byteLen : byteLen - 1);
    if (mod8 > 0) {
        uchar temp[32] = {0};
        for(int bit = b_start; bit < b_end; bit++) {
            temp[bit] = loaded_bits[bit * byteLen + b];
        }
        if (mod8 >= 1) {
            int32_t adder_0 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_0 |= ((uint32_t)((temp[bit] & (0x80 >> 0)) >> (7 - 0))) << (31 - bit);
            }
            quant_inds[i] += (adder_0 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 2) {
            int32_t adder_1 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_1 |= ((uint32_t)((temp[bit] & (0x80 >> 1)) >> (7 - 1))) << (31 - bit);
            }
            quant_inds[i + 1] += (adder_1 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 3) {
            int32_t adder_2 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_2 |= ((uint32_t)((temp[bit] & (0x80 >> 2)) >> (7 - 2))) << (31 - bit);
            }
            quant_inds[i + 2] += (adder_2 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 4) {
            int32_t adder_3 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_3 |= ((uint32_t)((temp[bit] & (0x80 >> 3)) >> (7 - 3))) << (31 - bit);
            }
            quant_inds[i + 3] += (adder_3 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 5) {
            int32_t adder_4 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_4 |= ((uint32_t)((temp[bit] & (0x80 >> 4)) >> (7 - 4))) << (31 - bit);
            }
            quant_inds[i + 4] += (adder_4 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 6) {
            int32_t adder_5 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_5 |= ((uint32_t)((temp[bit] & (0x80 >> 5)) >> (7 - 5))) << (31 - bit);
            }
            quant_inds[i + 5] += (adder_5 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }
        if (mod8 >= 7) {
            int32_t adder_6 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_6 |= ((uint32_t)((temp[bit] & (0x80 >> 6)) >> (7 - 6))) << (31 - bit);
            }
            quant_inds[i + 6] += (adder_6 ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }

    }
}

inline void add_to_quant_ori(aligned_vector<int32_t>& quant_inds, uchar* loaded_bits, int b_start, int b_end) {
    size_t intLen = quant_inds.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    int mod8 = intLen % 8;
    // #pragma omp parallel for
    for (size_t b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++) {

        size_t i = b * 8;
        uint32_t adder[8] = {0};

        for(int bit = b_start; bit < b_end; bit++) {
            uchar loaded = loaded_bits[bit * byteLen + b];
            uint32_t mask = 1U << (31 - bit);

            adder[0] |= (loaded & (0x80)) ? mask : 0;
            adder[1] |= (loaded & (0x40)) ? mask : 0;
            adder[2] |= (loaded & (0x20)) ? mask : 0;
            adder[3] |= (loaded & (0x10)) ? mask : 0;
            adder[4] |= (loaded & (0x08)) ? mask : 0;
            adder[5] |= (loaded & (0x04)) ? mask : 0;
            adder[6] |= (loaded & (0x02)) ? mask : 0;
            adder[7] |= (loaded & (0x01)) ? mask : 0;
        }

        // for(int ii = 0; ii < 8; ii++) {
        //     quant_inds[i + ii] |= adder[ii];
        // }

        quant_inds[i] |= adder[0];
        quant_inds[i + 1] += adder[1];
        quant_inds[i + 2] += adder[2];
        quant_inds[i + 3] += adder[3];
        quant_inds[i + 4] += adder[4];
        quant_inds[i + 5] += adder[5];
        quant_inds[i + 6] += adder[6];
        quant_inds[i + 7] += adder[7];
    }

    size_t i = intLen / 8 * 8, b = (mod8 == 0 ? byteLen : byteLen - 1);
    if (mod8 > 0) {
        uchar temp[32] = {0};
        for(int bit = b_start; bit < b_end; bit++) {
            temp[bit] = loaded_bits[bit * byteLen + b];
        }
        if (mod8 >= 1) {
            int32_t adder_0 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_0 |= ((uint32_t)((temp[bit] & (0x80 >> 0)) >> (7 - 0))) << (31 - bit);
            }
            quant_inds[i] |= adder_0;
        }
        if (mod8 >= 2) {
            int32_t adder_1 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_1 |= ((uint32_t)((temp[bit] & (0x80 >> 1)) >> (7 - 1))) << (31 - bit);
            }
            quant_inds[i + 1] += adder_1;
        }
        if (mod8 >= 3) {
            int32_t adder_2 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_2 |= ((uint32_t)((temp[bit] & (0x80 >> 2)) >> (7 - 2))) << (31 - bit);
            }
            quant_inds[i + 2] += adder_2;
        }
        if (mod8 >= 4) {
            int32_t adder_3 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_3 |= ((uint32_t)((temp[bit] & (0x80 >> 3)) >> (7 - 3))) << (31 - bit);
            }
            quant_inds[i + 3] += adder_3;
        }
        if (mod8 >= 5) {
            int32_t adder_4 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_4 |= ((uint32_t)((temp[bit] & (0x80 >> 4)) >> (7 - 4))) << (31 - bit);
            }
            quant_inds[i + 4] += adder_4;
        }
        if (mod8 >= 6) {
            int32_t adder_5 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_5 |= ((uint32_t)((temp[bit] & (0x80 >> 5)) >> (7 - 5))) << (31 - bit);
            }
            quant_inds[i + 5] += adder_5;
        }
        if (mod8 >= 7) {
            int32_t adder_6 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_6 |= ((uint32_t)((temp[bit] & (0x80 >> 6)) >> (7 - 6))) << (31 - bit);
            }
            quant_inds[i + 6] += adder_6;
        }

    }
}

};      // namespace SZ3
#endif  // SZ3_BYTEUTIL_HPP
