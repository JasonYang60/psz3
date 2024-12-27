#ifndef SZ3_SZALGOINTERPPROGRESSIVE_HPP
#define SZ3_SZALGOINTERPPROGRESSIVE_HPP

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
    size_t SZ_compress_Interp_nega_corel(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
        assert(N == conf.N);
        assert(conf.cmprAlgo == ALGO_INTERP);
        calAbsErrorBound(conf, data);
        
        auto sz = make_compressor_sz_progressive<T, N>(
            make_decomposition_interpolation<T, N>(conf,
                                                //    LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
                                                   NegabinaryQuantizer<T>(conf.absErrorBound)),
            BitplaneEncoder<int>(),
            Lossless_zstd());
        return sz->compress(conf, data, cmpData, cmpCap);
//        return cmpData;
    }
    
    template<class T, uint N>
    void SZ_decompress_Interp_nega_corel(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
        assert(conf.cmprAlgo == ALGO_INTERP);
        auto cmpDataPos = cmpData;
        auto sz = make_compressor_sz_progressive<T, N>(
            make_decomposition_interpolation<T, N>(conf,
                                                //    LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
                                                   NegabinaryQuantizer<T>(conf.absErrorBound)),

            BitplaneEncoder<int>(),
            Lossless_zstd());
        sz->decompress(conf, cmpDataPos, cmpSize, decData);
    }
}
#endif
