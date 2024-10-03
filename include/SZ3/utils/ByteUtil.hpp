//
// Created by Kai Zhao on 1/28/21.
//

#ifndef SZ3_BYTEUTIL_HPP
#define SZ3_BYTEUTIL_HPP

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

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

inline int16_t bytesToInt16_bigEndian(unsigned char *bytes) {
    int16_t temp = 0;
    int16_t res = 0;

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
inline void encode_int_2bits(const std::vector<int> &data, uchar *&c) {

    size_t intLen = data.size();
    size_t byteLen = intLen * 2 / 8 + (intLen % 4 == 0 ? 0 : 1);

    write(intLen, c);
    write(byteLen, c);

    size_t b, i = 0;
    int mod4 = intLen % 4;
    for (b = 0; b < (mod4 == 0 ? byteLen : byteLen - 1); b++, i += 4) {
        c[b] = (data[i] << 6) | (data[i + 1] << 4) | (data[i + 2] << 2) | data[i + 3];
    }
    if (mod4 > 0) {
        if (mod4 == 1) {
            c[b] = (data[i] << 6);
        } else if (mod4 == 2) {
            c[b] = (data[i] << 6) | (data[i + 1] << 4);
        } else if (mod4 == 3) {
            c[b] = (data[i] << 6) | (data[i + 1] << 4) | (data[i + 2] << 2);
        }
    }
    c += byteLen;
}


std::vector<int> decode_int_2bits(const uchar *&c, size_t &remaining_length) {
    size_t byteLen, intLen;
    read(intLen, c, remaining_length);
    read(byteLen, c, remaining_length);
    std::vector<int> ints(intLen);
    size_t i = 0, b = 0;

    int mod4 = intLen % 4;
    for (; b < (mod4 == 0 ? byteLen : byteLen - 1); b++, i += 4) {
        ints[i] = (c[b] & 0xC0) >> 6;
        ints[i + 1] = (c[b] & 0x30) >> 4;
        ints[i + 2] = (c[b] & 0x0C) >> 2;
        ints[i + 3] = c[b] & 0x03;
    }
    if (mod4 > 0) {
        if (mod4 >= 1) {
            ints[i] = (c[b] & 0xC0) >> 6;
        }
        if (mod4 >= 2) {
            ints[i + 1] = (c[b] & 0x30) >> 4;
        }
        if (mod4 >= 3) {
            ints[i + 2] = (c[b] & 0x0C) >> 2;
        }
    }
    c += byteLen;
    remaining_length -= byteLen;
    return ints;
}


inline void encode_int_1bit(const std::vector<int> &data, uchar *&c) {

    size_t intLen = data.size();
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    // write(intLen, c);
    // write(byteLen, c);

    size_t b, i = 0;
    int mod8 = intLen % 8;
    for (b = 0; b < (mod8 == 0 ? byteLen : byteLen - 1); b++, i += 8) {
        c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4)
                | (data[i + 4] << 3) | (data[i + 5] << 2) | (data[i + 6] << 1) | (data[i + 7]);
    }
    if (mod8 > 0) {
        if (mod8 == 1) {
            c[b] = (data[i] << 7);
        } else if (mod8 == 2) {
            c[b] = (data[i] << 7) | (data[i + 1] << 6);
        } else if (mod8 == 3) {
            c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5);
        } else if (mod8 == 4) {
            uchar temp = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4);
            c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4);
        } else if (mod8 == 5) {
            c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4)
                    | (data[i + 4] << 3);
        } else if (mod8 == 6) {
            c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4)
                    | (data[i + 4] << 3) | (data[i + 5] << 2);
        } else if (mod8 == 7) {
            c[b] = (data[i] << 7) | (data[i + 1] << 6) | (data[i + 2] << 5) | (data[i + 3] << 4)
                    | (data[i + 4] << 3) | (data[i + 5] << 2) | (data[i + 6] << 1);
        }
    }
    c += byteLen;
}

std::vector<int> decode_int_1bit(const uchar *&c, size_t &remaining_length, size_t intLen) {
    // size_t byteLen, intLen;
    // read(intLen, c, remaining_length);
    // read(byteLen, c, remaining_length);
    size_t byteLen = intLen / 8 + (intLen % 8 == 0 ? 0 : 1);

    std::vector<int> ints(intLen);
    size_t i = 0, b = 0;

    int mod8 = intLen % 8;
    for (; b < (mod8 == 0 ? byteLen : byteLen - 1); b++, i += 8) {
        ints[i] = (c[b] & 0x80) >> 7;
        ints[i + 1] = (c[b] & 0x40) >> 6;
        ints[i + 2] = (c[b] & 0x20) >> 5;
        ints[i + 3] = (c[b] & 0x10) >> 4;
        ints[i + 4] = (c[b] & 0x08) >> 3;
        ints[i + 5] = (c[b] & 0x04) >> 2;
        ints[i + 6] = (c[b] & 0x02) >> 1;
        ints[i + 7] = (c[b] & 0x01);
    }
    if (mod8 > 0) {
        if (mod8 >= 1) {
            ints[i] = (c[b] & 0x80) >> 7;
        }
        if (mod8 >= 2) {
            ints[i + 1] = (c[b] & 0x40) >> 6;
        }
        if (mod8 >= 3) {
            ints[i + 2] = (c[b] & 0x20) >> 5;
        }
        if (mod8 >= 4) {
            ints[i + 3] = (c[b] & 0x10) >> 4;
        }
        if (mod8 >= 5) {
            ints[i + 4] = (c[b] & 0x08) >> 3;
        }
        if (mod8 >= 6) {
            ints[i + 5] = (c[b] & 0x04) >> 2;
        }
        if (mod8 >= 7) {
            ints[i + 6] = (c[b] & 0x02) >> 1;
        }

    }
    c += byteLen;
    remaining_length -= byteLen;
    return ints;
}

};      // namespace SZ3
#endif  // SZ3_BYTEUTIL_HPP
