//
// Created by Kai Zhao on 4/21/20.
//

#ifndef SZ_LOSSLESS_ZSTD_HPP
#define SZ_LOSSLESS_ZSTD_HPP

#include "zstd.h"
#include "SZ3/def.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/lossless/Lossless.hpp"

namespace SZ3 {
    class Lossless_zstd : public concepts::LosslessInterface {
     
     public:
        Lossless_zstd() = default;
        
        Lossless_zstd(int comp_level) : compression_level(comp_level) {};

        size_t compress(uchar *src, size_t srcSize, uchar *dst) {
            // size_t estimatedCompressedSize = srcSize < 100 ? 200 : srcSize * 1.2;
            // uchar *dstPos = dst;
            size_t dstSize = ZSTD_compressBound(srcSize);

            // write(srcSize, dstPos);
  
            size_t const countSize = ZSTD_compress(dst, dstSize, src, srcSize, compression_level);
            if (ZSTD_isError(countSize)) {
                assert(!ZSTD_isError(countSize));
            }
            return countSize;
//             size_t outSize = ZSTD_compress(dataOutPos, estimatedCompressedSize,
//                                            dataIn, inSize, compression_level);
//             outSize += sizeof(size_t);
// //            printf("[ZSTD] ratio = %.2f inSize = %lu outSize = %lu\n", inSize * 1.0 / outSize, inSize, outSize);
//             return outSize;
        }

        size_t compress(uchar *src, size_t srcLen, uchar *dst, size_t dstCap) {
//            size_t estimatedCompressedSize = std::max(size_t(srcLen * 1.2), size_t(400));
//            uchar *compressBytes = new uchar[estimatedCompressedSize];
//            uchar *dstPos = dst;
//            write(srcLen, dstPos);
//            if (dstCap < srcLen) {
//                throw std::invalid_argument(
//                    "dstCap not large enough for zstd");
//            }
            return ZSTD_compress(dst, dstCap, src, srcLen, compression_level);
//            dstLen += sizeof(size_t);
//            return compressBytes;
        }
        
        size_t decompress(const uchar *src, const size_t srcLen, uchar *dst, size_t dstCap) {
//            const uchar *dataPos = data;
//            size_t dataLength = 0;
//            read(dataLength, dataPos, compressedSize);

//            uchar *oriData = new uchar[dataLength];
            size_t const countSize = ZSTD_decompress(dst, dstCap, src, srcLen);
            if (ZSTD_isError(countSize)) {
                assert(!ZSTD_isError(countSize));
            }
            return countSize;
//            compressedSize = dataLength;
//            return oriData;
        }
        // uchar *decompress(const uchar *src, size_t &srcSize,) {
        //     const uchar *dataPos = data;
        //     size_t dataLength = 0;
        //     read(dataLength, dataPos, compressedSize);

        //     // uchar *oriData = new uchar[dataLength];
        //     size_t const countSize = ZSTD_decompress(, dataLength, src, srcSize);
        //     ZSTD_decompress(oriData, dataLength, dataPos, compressedSize);
        //     compressedSize = dataLength;
        //     return oriData;
        // }
        // void postcompress_data(uchar *data) {
        //     delete[] data;
        // }
        // void postdecompress_data(uchar *data) {
        //     delete[] data;
        // }
     private:
        int compression_level = 3;  //default setting of level is 3
    };
}
#endif //SZ_LOSSLESS_ZSTD_HPP
