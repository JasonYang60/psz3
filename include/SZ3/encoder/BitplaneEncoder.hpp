#ifndef _SZ_BITPLANE_ENCODER_HPP
#define _SZ_BITPLANE_ENCODER_HPP

#include "Encoder.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/utils/ByteUtil.hpp"
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
            aligned_vector<int32_t> binsCopy(num);

            for(int i = 0; i < num; i++) {
                binsCopy[i] = bins[i];
            }

            for (auto &a : binsCopy) {
                a = toNega(a);
            }

            toCorel(binsCopy);

            uchar* outbytes = bitTranspose8inverse(binsCopy);
            int totalSize = (num + 7) / 8 * 32;
            // printf("total size = %d\n", (int)totalSize);


            memcpy(bytes, outbytes, totalSize);
            bytes += totalSize;

            return totalSize;
            
        };

        void postprocess_encode() {};

        void preprocess_decode() {};

        std::vector<T> decode(const uchar *&bytes, size_t targetLength) {
            targetLength = num;
            size_t length = (targetLength + 7) / 8;
            uchar* bytesCopy = new uchar[length * 32];
            // printf("length * 32 = %d\n", (int)length * 32);

            memcpy(bytesCopy, bytes, length * 32);

            aligned_vector<T> bins(targetLength, 0);
            // int length = (num + 7) / 8;

            for(int bits = 2; bits < 32; bits++) {
                uchar * second_last_bit = bytesCopy + (bits - 2) * length;
                uchar * last_bit = bytesCopy + (bits - 1) * length; 
                uchar * thisBit = bytesCopy + bits * length;
                for(int i = 0; i < length; i++) {
                    thisBit[i] ^= last_bit[i] ^ second_last_bit[i];
                }
            }



            // printf("haha\n");
            uchar a = (bytesCopy[31 * length]);

            add_to_quant_ori(bins, bytesCopy, 0, 32);
            std::vector<T> out(targetLength);
            for(int i = 0; i < targetLength; i++) {
                out[i] = fromNega(bins[i]);
            }
            return out;
        };

        void postprocess_decode() {};

        void save(uchar *&c) {
            write(num, c);
        };


        void load(const uchar *&c, size_t &remaining_length) {
            read(num, c, remaining_length);
        };

    private:
        int num = 0;
        // uint32_t table = 0;
        std::vector<T> bins;
        

        void toCorel(aligned_vector<int32_t>& quants) {
            const int sz = (int)quants.size();
            // #pragma omp parallel for
            for(int i = 0; i < sz; i++) {
                // uint32_t qt = (uint32_t) quants[i];
                // uint32_t sel = qt >> 1;
                // uint32_t pred = (tab_1 & sel) | (tab_0 & ~sel);
                // qt ^= pred;
                // quants[i] = qt;

                uint32_t temp = (uint32_t)quants[i];
                temp ^= temp >> 1;
                quants[i] ^= temp >> 1;
                // quants[i] ^= (((uint32_t)quants[i]) >> 1);
            }
        }
    
        inline void fromCorel(const uint32_t tab, T& result, T last, int b) {
            if(b > 0) {
                result ^= (((last & 1) ? (tab & (1 << (15 - b))) : ((tab & (1 << (31 - b))) >> 16)) >> (15 - b));
            }
        }

        inline int toNega(int quant) {
            return ((int32_t) quant + (uint32_t) 0xaaaaaaaau) ^ (uint32_t) 0xaaaaaaaau;
        }

        inline int fromNega(int nega_quant) {
            return (((uint32_t) nega_quant) ^ 0xaaaaaaaau) - 0xaaaaaaaau;
        }

    };
}
#endif
