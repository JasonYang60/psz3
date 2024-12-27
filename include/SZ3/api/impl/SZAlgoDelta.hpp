#ifndef SZ3_SZALGODELTA_HPP
#define SZ3_SZALGODELTA_HPP

#include "SZ3/decomposition/InterpolationDecomposition.hpp"
#include "SZ3/compressor/specialized/SZBlockInterpolationCompressor.hpp"
#include "SZ3/compressor/SZProgressiveCompressor.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "SZ3/quantizer/NegabinaryQuantizer.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/Statistic.hpp"
#include "SZ3/utils/Extraction.hpp"
#include "SZ3/utils/QuantOptimizatioin.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/api/impl/SZAlgoLorenzoReg.hpp"
#include "SZ3/encoder/BitplaneEncoder.hpp"
#include <cmath>
#include <memory>

namespace SZ3 {
    template<class T, uint N>
    size_t SZ_compress_delta(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
        size_t total = 0;

        std::vector<double> targetEB = { 1e-2, 1e-4};
        T* dataCopy = new T[conf.num];
        T range = data_range(data, conf.num);
        memcpy(dataCopy, data, conf.num * sizeof(T));

        for(int i = 0; i < targetEB.size(); i++) {

            conf.relErrorBound = targetEB[i];
            conf.absErrorBound = conf.relErrorBound * range;

            
            auto sz = make_compressor_sz_generic<T, N>(
                make_decomposition_interpolation<T, N>(conf,
                                                    LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
                HuffmanEncoder<int>(),
                Lossless_zstd());
            if(i == 0){
            total += sz->compress(conf, data, cmpData, cmpCap);
            }
            T temp = 0;
            for(int i = 0; i < conf.num; i++) {
                temp = data[i];
                data[i] = data[i] - dataCopy[i];
                dataCopy[i] = temp;
            }
        }

        delete []dataCopy;
        return total;
        
    }

    template<class T, uint N>
    void SZ_decompress_delta(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
        auto cmpDataPos = cmpData;
        for(int i = 0; i < 1; i++) {
            auto sz = make_compressor_sz_generic<T, N>(
                make_decomposition_interpolation<T, N>(conf,
                                                    LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),

                HuffmanEncoder<int>(),
                Lossless_zstd());
            sz->decompress(conf, cmpDataPos, cmpSize, decData);
        }
    }
}
#endif
