#ifndef SZ_COMPRESSOR_TYPE_PROGRESSIVE_HPP
#define SZ_COMPRESSOR_TYPE_PROGRESSIVE_HPP

#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/decomposition/Decomposition.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/def.hpp"
#include <cstring>

/**
 * SZProgressiveCompressor glues together decomposition, encoder, and lossless modules to form the compression pipeline
 * it doesn't contains the logic to iterate through the input data. The logic is handled inside decomposition
 */

namespace SZ3 {
    template<class T, uint N, class Decomposition, class Encoder, class Lossless>
    class SZProgressiveCompressor : public concepts::CompressorInterface<T> {
    public:


        SZProgressiveCompressor(Decomposition decomposition, Encoder encoder, Lossless lossless) :
                decomposition(decomposition), encoder(encoder), lossless(lossless) {
            static_assert(std::is_base_of<concepts::DecompositionInterface<T, N>, Decomposition>::value,
                          "must implement the frontend interface");
            static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");
        }

        size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) {

            std::vector<int> quant_inds = decomposition.compress(conf, data);
            size_t bufferSize = std::max<size_t>(1000, 1.2 * (decomposition.size_est() + encoder.size_est() + sizeof(T) * quant_inds.size()));
            auto buffer = (uchar *) malloc(bufferSize);
            uchar *buffer_pos = buffer;

            decomposition.save(buffer_pos);

            auto total = compress_encode_and_lossless(quant_inds, cmpData, cmpCap, buffer_pos, buffer);

            free(buffer);

            return total;
        }

        T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) {
            auto quant_inds = decompress_lossless_and_decode(conf.num, cmpData, cmpSize);

            decomposition.decompress(conf, quant_inds, decData);
            return decData;
        }


    private:
        Decomposition decomposition;
        Encoder encoder;
        Lossless lossless;

        size_t compress_encode_and_lossless(std::vector<int>& quant_inds, uchar *cmpData, size_t cmpCap, uchar* buffer_pos, uchar* buffer) {
            
            encoder.preprocess_encode(quant_inds, decomposition.get_radius() * 2);
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, buffer_pos);
            encoder.postprocess_encode();

            auto cmpSize = lossless.compress(buffer, buffer_pos - buffer, cmpData, cmpCap);
            return cmpSize;
        }

        std::vector<int> decompress_lossless_and_decode(size_t num, uchar const *cmpData, size_t cmpSize) {
            size_t bufferCap = num * sizeof(T);
            auto buffer = (uchar *) malloc(bufferCap);
            lossless.decompress(cmpData, cmpSize, buffer, bufferCap);

            size_t remaining_length = bufferCap;
            uchar const *buffer_pos = buffer;

            decomposition.load(buffer_pos, remaining_length);
            encoder.load(buffer_pos, remaining_length);
            
            auto quant_inds = encoder.decode(buffer_pos, num);
            encoder.postprocess_decode();

            free(buffer);
            return quant_inds;
        }

        std::vector<std::vector<int>> splitBySegments(
            const std::vector<int>& quant, 
            const std::vector<size_t>& segments)
        {
            std::vector<std::vector<int>> result;
            result.reserve(segments.size());

            size_t currentPos = 0;
            
            for (auto len : segments)
            {
                if (currentPos + len > quant.size()) 
                {
                    throw std::out_of_range("Segment length exceeds the size of 'quant'.");
                }

                auto startIter = quant.begin() + currentPos;
                auto endIter   = quant.begin() + currentPos + len;
                
                result.emplace_back(startIter, endIter);
                
                currentPos += len;
            }

            return result;
        }
    
        std::vector<int> concatenateSegments(const std::vector<std::vector<int>>& segments)
        {
            size_t totalSize = 0;
            for (const auto& seg : segments)
            {
                totalSize += seg.size();
            }
            
            std::vector<int> result;
            result.reserve(totalSize);

            for (const auto& seg : segments)
            {
                result.insert(result.end(), seg.begin(), seg.end());
            }
            
            return result;
        }
    };

    template<class T, uint N, class Decomposition, class Encoder, class Lossless>
    std::shared_ptr<SZProgressiveCompressor<T, N, Decomposition, Encoder, Lossless>>
    make_compressor_sz_progressive(Decomposition decomposition, Encoder encoder, Lossless lossless) {
        return std::make_shared<SZProgressiveCompressor<T, N, Decomposition, Encoder, Lossless>>(decomposition, encoder, lossless);
    }

}
#endif
