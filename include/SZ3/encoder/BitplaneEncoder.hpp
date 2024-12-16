#ifndef _SZ_BITPLANE_ENCODER_HPP
#define _SZ_BITPLANE_ENCODER_HPP

#include "Encoder.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/def.hpp"
#include <vector>

namespace SZ3 {

    template<class T>
    class BitplaneEncoder : public concepts::EncoderInterface<T> {
    public:

        ~BitplaneEncoder() = default;

        void preprocess_encode(const std::vector<T> &bins, int stateNum) {
            num = bins.size();
        };

        size_t encode(const std::vector<T> &bins, uchar *&bytes) {
            std::vector<int> binsCopy = bins;

            uint32_t table = predict_table(bins);

            write(table, bytes);
            toCorel(table, binsCopy);

            std::vector<int> result;
            result.reserve(bins.size());

            for (int i = 0; i < 16; ++i) {
                for (auto x : binsCopy) {
                    int top_bit = (x & 0x00008000) ? 1 : 0;
                    result.push_back(top_bit);
                }

                encode_int_1bit(result, bytes);
                result.clear();
                for (auto &x : binsCopy) {
                    x <<= 1;
                }
            }
            return 0;
        };

        void postprocess_encode() {};

        void preprocess_decode() {};

        std::vector<T> decode(const uchar *&bytes, size_t targetLength) {
            return bins;
        };

        void postprocess_decode() {};

        void save(uchar *&c) {
            write(num, c);
        };


        void load(const uchar *&c, size_t &remaining_length) {
            read(num, c, remaining_length);
            read(table, c, remaining_length);
            std::vector<T> result;
            result.resize(num);
            bins.clear();
            bins.resize(num, 0);
            for(int i = 0; i < 16; i++) {
                result = decode_int_1bit(c, remaining_length, num);
                for(int j = 0; j < num; j++) {
                    fromCorel(table, result[j], bins[j], i);
                    bins[j] <<= 1;
                    bins[j] += result[j];
                }
            }
        };

    private:
        int num = 0;
        uint32_t table = 0;
        std::vector<T> bins;
        
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

        uint32_t predict_table(const std::vector<int>& bins) {
            int cnt_zero_zero[15] = {0};
            int cnt_zero_one[15] = {0};
            int cnt_one_zero[15] = {0};
            int cnt_one_one[15] = {0};
            for(int i = 0; i < bins.size(); i++) {
                uint32_t qt = bins[i] << 16;
                
                for(int b = 0; b < 16 - 1; b++){
                    cnt_zero_zero[b] += (qt & (uint32_t)0xc0000000u) == (uint32_t)0x00000000u;
                    cnt_zero_one[b] += (qt & (uint32_t)0xc0000000u) == (uint32_t)0x40000000u;
                    cnt_one_zero[b] += (qt & (uint32_t)0xc0000000u) == (uint32_t)0x80000000u;
                    cnt_one_one[b] += (qt & (uint32_t)0xc0000000u) == (uint32_t)0xc0000000u;
                    qt = qt << 1;
                }
            }
            uint32_t output = 0;
            for(int b = 0; b < 15; b++) {
                output = ((uint32_t)(cnt_zero_zero[b] < cnt_zero_one[b]) << (30 - b)) | output;
            }
            for(int b = 16; b < 31; b++) {
                output = ((uint32_t)(cnt_one_zero[b - 16] < cnt_one_one[b - 16]) << (30 - b)) | output;
            }
            return output;
        }

        void toCorel(const uint32_t tab, std::vector<int>& bins) {
            int sz = bins.size();
            for(int i = 0; i < sz; i++){
                uint32_t qt = bins[i];
                for(int b = 15; b >= 1; b--){
                    qt = ((qt & (1 << (16 - b))) ? (tab & (1 << (15 - b))) : ((tab & (1 << (31 - b))) >> 16)) ^ qt;
                }
                bins[i] = qt;
            }
        }
    
        inline void fromCorel(const uint32_t tab, T& result, T last, int b) {
            if(b > 0) {
                result ^= (((last & 1) ? (tab & (1 << (15 - b))) : ((tab & (1 << (31 - b))) >> 16)) >> (15 - b));
            }
        }
        // void invert_table(const uint32_t tab, std::vector<int>& quant_ind_truncated, int b) {
        //     int sz = quant_ind_truncated.size();
        //     assert(sz == quant_inds.size());
            
        //     if(b > 0) {
        //         for(int i = 0; i < sz; i++){
        //             quant_ind_truncated[i] = quant_ind_truncated[i] ^ ((last_bit[lid][i] ? (tab & (1 << (15 - b))) : ((tab & (1 << (31 - b))) >> 16)) >> (15 - b));
        //         }
        //     }
        // }
    };
}
#endif
