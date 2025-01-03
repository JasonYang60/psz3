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

inline uchar* bitTranspose8(aligned_vector<int32_t> &in)
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
        uint32_t mask = 1 << bit;
        for(size_t b = 0; b < nBlocks; b++) {
            size_t baseIn = b * blockSize;
            uint32_t in_0 = (in[baseIn + 0] & mask) >> bit;
            uint32_t in_1 = (in[baseIn + 1] & mask) >> bit;
            uint32_t in_2 = (in[baseIn + 2] & mask) >> bit;
            uint32_t in_3 = (in[baseIn + 3] & mask) >> bit;
            uint32_t in_4 = (in[baseIn + 4] & mask) >> bit;
            uint32_t in_5 = (in[baseIn + 5] & mask) >> bit;
            uint32_t in_6 = (in[baseIn + 6] & mask) >> bit;
            uint32_t in_7 = (in[baseIn + 7] & mask) >> bit;
            out[bit * nBlocks + b] = (in_0 << 7) | (in_1 << 6) | (in_2 << 5) | (in_3 << 4)
                | (in_4 << 3) | (in_5 << 2) | (in_6 << 1) | ((in_7 & 1u));
        }
    }

    return out;
}

inline uint64_t* bitTranspose64(aligned_vector<int32_t> &in)
{
    if (in.size() % 64 != 0) {
        int res = 64 - in.size() % 64;
        for(int i = 0; i < res; i++) { in.push_back(0); }
    }

    const size_t blockSize = 64;    // Process 64 integers per block
    const size_t bitsPerInt = 32;  // Each int has 32 bits
    size_t nBlocks = in.size() / blockSize;

    uint64_t* out = static_cast<uint64_t*>(::operator new(nBlocks * bitsPerInt * sizeof(uint64_t), std::align_val_t(256)));

    for (size_t bit = 0; bit < bitsPerInt; bit++) {
        uint32_t mask = 1u << bit;
        for (size_t b = 0; b < nBlocks; b++) {
            size_t baseIn = b * blockSize;
            uint32_t in_0 = (in[baseIn + 0] & mask) >> bit;
            uint32_t in_1 = (in[baseIn + 1] & mask) >> bit;
            uint32_t in_2 = (in[baseIn + 2] & mask) >> bit;
            uint32_t in_3 = (in[baseIn + 3] & mask) >> bit;
            uint32_t in_4 = (in[baseIn + 4] & mask) >> bit;
            uint32_t in_5 = (in[baseIn + 5] & mask) >> bit;
            uint32_t in_6 = (in[baseIn + 6] & mask) >> bit;
            uint32_t in_7 = (in[baseIn + 7] & mask) >> bit;
            uint32_t in_8 = (in[baseIn + 8] & mask) >> bit;
            uint32_t in_9 = (in[baseIn + 9] & mask) >> bit;
            uint32_t in_10 = (in[baseIn + 10] & mask) >> bit;
            uint32_t in_11 = (in[baseIn + 11] & mask) >> bit;
            uint32_t in_12 = (in[baseIn + 12] & mask) >> bit;
            uint32_t in_13 = (in[baseIn + 13] & mask) >> bit;
            uint32_t in_14 = (in[baseIn + 14] & mask) >> bit;
            uint32_t in_15 = (in[baseIn + 15] & mask) >> bit;
            uint32_t in_16 = (in[baseIn + 16] & mask) >> bit;
            uint32_t in_17 = (in[baseIn + 17] & mask) >> bit;
            uint32_t in_18 = (in[baseIn + 18] & mask) >> bit;
            uint32_t in_19 = (in[baseIn + 19] & mask) >> bit;
            uint32_t in_20 = (in[baseIn + 20] & mask) >> bit;
            uint32_t in_21 = (in[baseIn + 21] & mask) >> bit;
            uint32_t in_22 = (in[baseIn + 22] & mask) >> bit;
            uint32_t in_23 = (in[baseIn + 23] & mask) >> bit;
            uint32_t in_24 = (in[baseIn + 24] & mask) >> bit;
            uint32_t in_25 = (in[baseIn + 25] & mask) >> bit;
            uint32_t in_26 = (in[baseIn + 26] & mask) >> bit;
            uint32_t in_27 = (in[baseIn + 27] & mask) >> bit;
            uint32_t in_28 = (in[baseIn + 28] & mask) >> bit;
            uint32_t in_29 = (in[baseIn + 29] & mask) >> bit;
            uint32_t in_30 = (in[baseIn + 30] & mask) >> bit;
            uint32_t in_31 = (in[baseIn + 31] & mask) >> bit;
            uint32_t in_32 = (in[baseIn + 32] & mask) >> bit;
            uint32_t in_33 = (in[baseIn + 33] & mask) >> bit;
            uint32_t in_34 = (in[baseIn + 34] & mask) >> bit;
            uint32_t in_35 = (in[baseIn + 35] & mask) >> bit;
            uint32_t in_36 = (in[baseIn + 36] & mask) >> bit;
            uint32_t in_37 = (in[baseIn + 37] & mask) >> bit;
            uint32_t in_38 = (in[baseIn + 38] & mask) >> bit;
            uint32_t in_39 = (in[baseIn + 39] & mask) >> bit;
            uint32_t in_40 = (in[baseIn + 40] & mask) >> bit;
            uint32_t in_41 = (in[baseIn + 41] & mask) >> bit;
            uint32_t in_42 = (in[baseIn + 42] & mask) >> bit;
            uint32_t in_43 = (in[baseIn + 43] & mask) >> bit;
            uint32_t in_44 = (in[baseIn + 44] & mask) >> bit;
            uint32_t in_45 = (in[baseIn + 45] & mask) >> bit;
            uint32_t in_46 = (in[baseIn + 46] & mask) >> bit;
            uint32_t in_47 = (in[baseIn + 47] & mask) >> bit;
            uint32_t in_48 = (in[baseIn + 48] & mask) >> bit;
            uint32_t in_49 = (in[baseIn + 49] & mask) >> bit;
            uint32_t in_50 = (in[baseIn + 50] & mask) >> bit;
            uint32_t in_51 = (in[baseIn + 51] & mask) >> bit;
            uint32_t in_52 = (in[baseIn + 52] & mask) >> bit;
            uint32_t in_53 = (in[baseIn + 53] & mask) >> bit;
            uint32_t in_54 = (in[baseIn + 54] & mask) >> bit;
            uint32_t in_55 = (in[baseIn + 55] & mask) >> bit;
            uint32_t in_56 = (in[baseIn + 56] & mask) >> bit;
            uint32_t in_57 = (in[baseIn + 57] & mask) >> bit;
            uint32_t in_58 = (in[baseIn + 58] & mask) >> bit;
            uint32_t in_59 = (in[baseIn + 59] & mask) >> bit;
            uint32_t in_60 = (in[baseIn + 60] & mask) >> bit;
            uint32_t in_61 = (in[baseIn + 61] & mask) >> bit;
            uint32_t in_62 = (in[baseIn + 62] & mask) >> bit;
            uint32_t in_63 = (in[baseIn + 63] & mask) >> bit;

            uint64_t transposedBits = (static_cast<uint64_t>(in_0) << 63) | (static_cast<uint64_t>(in_1) << 62) |
                                      (static_cast<uint64_t>(in_2) << 61) | (static_cast<uint64_t>(in_3) << 60) |
                                      (static_cast<uint64_t>(in_4) << 59) | (static_cast<uint64_t>(in_5) << 58) |
                                      (static_cast<uint64_t>(in_6) << 57) | (static_cast<uint64_t>(in_7) << 56) |
                                      (static_cast<uint64_t>(in_8) << 55) | (static_cast<uint64_t>(in_9) << 54) |
                                      (static_cast<uint64_t>(in_10) << 53) | (static_cast<uint64_t>(in_11) << 52) |
                                      (static_cast<uint64_t>(in_12) << 51) | (static_cast<uint64_t>(in_13) << 50) |
                                      (static_cast<uint64_t>(in_14) << 49) | (static_cast<uint64_t>(in_15) << 48) |
                                      (static_cast<uint64_t>(in_16) << 47) | (static_cast<uint64_t>(in_17) << 46) |
                                      (static_cast<uint64_t>(in_18) << 45) | (static_cast<uint64_t>(in_19) << 44) |
                                      (static_cast<uint64_t>(in_20) << 43) | (static_cast<uint64_t>(in_21) << 42) |
                                      (static_cast<uint64_t>(in_22) << 41) | (static_cast<uint64_t>(in_23) << 40) |
                                      (static_cast<uint64_t>(in_24) << 39) | (static_cast<uint64_t>(in_25) << 38) |
                                      (static_cast<uint64_t>(in_26) << 37) | (static_cast<uint64_t>(in_27) << 36) |
                                      (static_cast<uint64_t>(in_28) << 35) | (static_cast<uint64_t>(in_29) << 34) |
                                      (static_cast<uint64_t>(in_30) << 33) | (static_cast<uint64_t>(in_31) << 32) |
                                      (static_cast<uint64_t>(in_32) << 31) | (static_cast<uint64_t>(in_33) << 30) |
                                      (static_cast<uint64_t>(in_34) << 29) | (static_cast<uint64_t>(in_35) << 28) |
                                      (static_cast<uint64_t>(in_36) << 27) | (static_cast<uint64_t>(in_37) << 26) |
                                      (static_cast<uint64_t>(in_38) << 25) | (static_cast<uint64_t>(in_39) << 24) |
                                      (static_cast<uint64_t>(in_40) << 23) | (static_cast<uint64_t>(in_41) << 22) |
                                      (static_cast<uint64_t>(in_42) << 21) | (static_cast<uint64_t>(in_43) << 20) |
                                      (static_cast<uint64_t>(in_44) << 19) | (static_cast<uint64_t>(in_45) << 18) |
                                      (static_cast<uint64_t>(in_46) << 17) | (static_cast<uint64_t>(in_47) << 16) |
                                      (static_cast<uint64_t>(in_48) << 15) | (static_cast<uint64_t>(in_49) << 14) |
                                      (static_cast<uint64_t>(in_50) << 13) | (static_cast<uint64_t>(in_51) << 12) |
                                      (static_cast<uint64_t>(in_52) << 11) | (static_cast<uint64_t>(in_53) << 10) |
                                      (static_cast<uint64_t>(in_54) << 9) | (static_cast<uint64_t>(in_55) << 8) |
                                      (static_cast<uint64_t>(in_56) << 7) | (static_cast<uint64_t>(in_57) << 6) |
                                      (static_cast<uint64_t>(in_58) << 5) | (static_cast<uint64_t>(in_59) << 4) |
                                      (static_cast<uint64_t>(in_60) << 3) | (static_cast<uint64_t>(in_61) << 2) |
                                      (static_cast<uint64_t>(in_62) << 1) | (static_cast<uint64_t>(in_63));

            out[bit * nBlocks + b] = transposedBits;
        }
    }

    return out;
}

// inline uchar* bitTranspose8(aligned_vector<int32_t> &in)
// {
//     if (in.size() % 8 != 0) {
//         int res = 8 - in.size() % 8;
//         for(int i = 0; i < res; i++) {in.push_back(0); }
//     }

//     const size_t blockSize = 8;     
//     const size_t bitsPerInt = 32;   
//     size_t nBlocks = in.size() / blockSize + 1;

//     uchar* out = static_cast<uchar*>(::operator new(nBlocks * bitsPerInt, std::align_val_t(256)));
//     // #pragma omp parallel for
//     for (size_t b = 0; b < nBlocks; b++) {
//         size_t baseIn = b * blockSize;
//         size_t baseOut = b * bitsPerInt;

//         int in_0 = in[baseIn + 0];
//         int in_1 = in[baseIn + 1];
//         int in_2 = in[baseIn + 2];
//         int in_3 = in[baseIn + 3];
//         int in_4 = in[baseIn + 4];
//         int in_5 = in[baseIn + 5];
//         int in_6 = in[baseIn + 6];
//         int in_7 = in[baseIn + 7];

//         for(int bit = 0; bit < bitsPerInt; bit++){
//             out[bit * nBlocks + b] = ((in_0 & 1u) << 7) | ((in_1 & 1u) << 6) | ((in_2 & 1u) << 5) | ((in_3 & 1u) << 4)
//                 | ((in_4 & 1u) << 3) | ((in_5 & 1u) << 2) | ((in_6 & 1u) << 1) | ((in_7 & 1u));
            
//             in_0 >>= 1;
//             in_1 >>= 1;
//             in_2 >>= 1;
//             in_3 >>= 1;
//             in_4 >>= 1;
//             in_5 >>= 1;
//             in_6 >>= 1;
//             in_7 >>= 1;
//         }
//     }

//     return out;
// }


inline void add_to_quant(aligned_vector<int32_t>& quant_inds, const aligned_vector<uchar>& bytes, int bitshift) {
    size_t intLen = quant_inds.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    int mod8 = intLen % 8;
    // #pragma omp parallel for
    for (size_t b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++) {
        size_t i = b * 8;
        uint32_t temp = bytes[b];
        // quant_inds[i] += ((temp & 0x80) >> 7) << bitshift;
        // quant_inds[i + 1] += ((temp & 0x40) >> 6) << bitshift;
        // quant_inds[i + 2] += ((temp & 0x20) >> 5) << bitshift;
        // quant_inds[i + 3] += ((temp & 0x10) >> 4) << bitshift;
        // quant_inds[i + 4] += ((temp & 0x08) >> 3) << bitshift;
        // quant_inds[i + 5] += ((temp & 0x04) >> 2) << bitshift;
        // quant_inds[i + 6] += ((temp & 0x02) >> 1) << bitshift;
        // quant_inds[i + 7] += ((temp & 0x01)) << bitshift;
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

inline void add_to_quant(aligned_vector<int32_t>& quant_inds, uchar* loaded_bits, int b_start, int b_end) {
    size_t intLen = quant_inds.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    int mod8 = intLen % 8;
    // #pragma omp parallel for
    for (size_t b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++) {
        // size_t i = b * 8;
        // uchar temp[32] = {0};
        // int32_t adder[8] = {0};
        // for(int bit = b_start; bit < b_end; bit++) {
        //     temp[bit] = loaded_bits[bit * byteLen + b];
        // }
        // for(int bit = b_start; bit < b_end; bit++) {
        //     for(int ii = 0; ii < 8; ii++) {
        //         adder[ii] |= ((uint32_t)((temp[bit] & (0x80 >> ii)) >> (7 - ii))) << (31 - bit);
        //     }
        // }
        // for(int ii = 0; ii < 8; ii++) {
        //     quant_inds[i + ii] |= adder[ii];
        // }
        size_t i = b * 8;
        int32_t adder[8] = {0};

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
        quant_inds[i] |= adder[0];
        quant_inds[i + 1] |= adder[1];
        quant_inds[i + 2] |= adder[2];
        quant_inds[i + 3] |= adder[3];
        quant_inds[i + 4] |= adder[4];
        quant_inds[i + 5] |= adder[5];
        quant_inds[i + 6] |= adder[6];
        quant_inds[i + 7] |= adder[7];


        // quant_inds[i + 1] += ((temp & 0x40) >> 6) << bitshift;
        // quant_inds[i + 2] += ((temp & 0x20) >> 5) << bitshift;
        // quant_inds[i + 3] += ((temp & 0x10) >> 4) << bitshift;
        // quant_inds[i + 4] += ((temp & 0x08) >> 3) << bitshift;
        // quant_inds[i + 5] += ((temp & 0x04) >> 2) << bitshift;
        // quant_inds[i + 6] += ((temp & 0x02) >> 1) << bitshift;
        // quant_inds[i + 7] += ((temp & 0x01)) << bitshift;
        // uint32_t masks[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
        // for (size_t j = 0; j < 8; ++j) {
        //     quant_inds[i + j] |= ((temp & masks[j]) >> (7 - j)) << bitshift;
        // }
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
            quant_inds[i + 1] |= adder_1;
        }
        if (mod8 >= 3) {
            int32_t adder_2 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_2 |= ((uint32_t)((temp[bit] & (0x80 >> 2)) >> (7 - 2))) << (31 - bit);
            }
            quant_inds[i + 2] |= adder_2;
        }
        if (mod8 >= 4) {
            int32_t adder_3 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_3 |= ((uint32_t)((temp[bit] & (0x80 >> 3)) >> (7 - 3))) << (31 - bit);
            }
            quant_inds[i + 3] |= adder_3;
        }
        if (mod8 >= 5) {
            int32_t adder_4 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_4 |= ((uint32_t)((temp[bit] & (0x80 >> 4)) >> (7 - 4))) << (31 - bit);
            }
            quant_inds[i + 4] |= adder_4;
        }
        if (mod8 >= 6) {
            int32_t adder_5 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_5 |= ((uint32_t)((temp[bit] & (0x80 >> 5)) >> (7 - 5))) << (31 - bit);
            }
            quant_inds[i + 5] |= adder_5;
        }
        if (mod8 >= 7) {
            int32_t adder_6 = 0;
            for(int bit = b_start; bit < b_end; bit++) {
                adder_6 |= ((uint32_t)((temp[bit] & (0x80 >> 6)) >> (7 - 6))) << (31 - bit);
            }
            quant_inds[i + 6] |= adder_6;
        }

    }
}


};      // namespace SZ3
#endif  // SZ3_BYTEUTIL_HPP
