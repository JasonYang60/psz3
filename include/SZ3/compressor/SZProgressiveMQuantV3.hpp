#ifndef _SZ_SZ_PROG_INTERPOLATION_MULTILEVEL_QUANTIZATION_HPP
#define _SZ_SZ_PROG_INTERPOLATION_MULTILEVEL_QUANTIZATION_HPP

#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Interpolators.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/utils/ByteUtil.hpp"
#include "SZ3/utils/ska_hash/unordered_map.hpp"
#include "SZ3/utils/Verification.hpp"
#include "SZ3/def.hpp"
#include <cstring>
#include <cmath>

namespace SZ3 {
    template<class T, uint N, class Quantizer, class Encoder, class Lossless>
    class SZProgressiveMQuant {
    public:


        SZProgressiveMQuant(Quantizer quantizer, Encoder encoder, Lossless lossless,
                            const std::array<size_t, N> dims,
                            int interpolator,
                            int direction_id_,
                            size_t interp_dim_limit,
                            // int level_progressive_,
                            int layers_,
                            size_t block_size_
                            //int level_fill_
                            ) :
                quantizer(quantizer), encoder(encoder), lossless(lossless),
                global_dimensions(dims),
                interpolators({"linear", "cubic"}),
                interpolator_id(interpolator),
                interp_dim_limit(interp_dim_limit),
                // level_progressive(level_progressive_),
                layers(layers_),
                block_size(block_size_)
//                level_fill(level_fill_)
        {
            static_assert(std::is_base_of<concepts::QuantizerInterface<T>, Quantizer>::value,
                          "must implement the quatizer interface");
//            static_assert(std::is_base_of<concepts::EncoderInterface<>, Encoder>::value,
//                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");

            assert(interp_dim_limit % 2 == 0 &&
                   "Interpolation dimension should be even numbers to avoid extrapolation");
            num_elements = 1;
            levels = -1;
            for (int i = 0; i < N; i++) {
                if (levels < ceil(log2(dims[i]))) {
                    levels = (uint) ceil(log2(dims[i]));
                }
                num_elements *= dims[i];
                global_begin[i] = 0;
                global_end[i] = global_dimensions[i] - 1;
            }

            dim_offsets[N - 1] = 1;
            for (int i = N - 2; i >= 0; i--) {
                dim_offsets[i] = dim_offsets[i + 1] * global_dimensions[i + 1];
            }
            set_directions_and_stride(direction_id_);

            level_progressive = levels;

            
        }

        T *decompress(uchar const *lossless_data, T *data, std::vector<T> &targetEBs) {
            setupLayers(data);
            printf("range = %f\n", range);
            for(auto &eb : targetEBs){
                eb *= range;    // relative error bound
            }
            T *dec_data = new T[num_elements];
            if(targetEBs.empty()){
                std::cout << "[error]target error bound is empty." << std::endl;
                return dec_data;
            }
            std::cout << std::endl;
            std::cout << "-------- error bound = " << targetEBs[0] << " --------" << std::endl;
            
            decompress(lossless_data, data, dec_data, targetEBs[0], 0);
            printf("[Log] retrieved = %.3f%% %lu\n", retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size);
            std::cout << "-------- compression ratio = " << (num_elements * sizeof(T)) * 1.0/ retrieved_size  << " --------" << std::endl;
            
            for(int i = 1; i < targetEBs.size(); i++) {
                std::cout << std::endl;
                std::cout << "-------- error bound = "  << targetEBs[i] << " --------" << std::endl;
                decompress(lossless_data, data, dec_data, targetEBs[i], targetEBs[i - 1]);
                printf("[Log] retrieved = %.3f%% %lu\n", retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size);
                std::cout << "-------- compression ratio = " << (num_elements * sizeof(T)) * 1.0 / retrieved_size << " --------" << std::endl;
            }
            return dec_data;
        }

        void *decompress(uchar const *lossless_data, T *data, T *dec_data, const T targetErrorBound, T lastEB) {
            // if(targetErrorBound >= lastEB){
            //     return dec_data;
            // }
            int lsize = N * level_progressive, bsize = bitgroup.size();
            std::vector<int> bsum(lsize, 0), bdelta(lsize, bsize);
            uchar const * lossless_data_pos = lossless_data;

            std::vector<size_t> levelSize(lsize, 0);
            std::vector<size_t> lossless_size;
            uchar const * cmp_data_pos = nullptr;
            loadcfg(lossless_data_pos, levelSize, lossless_size, cmp_data_pos);

            lossless_size.erase(lossless_size.begin());
            // lossless_size.erase(lossless_size.end());
            lossless_size.pop_back(); 
            // int progressive_layer_old, progressive_layer_new = 0;
            std::vector<std::vector<int>> bitGroupOfLayer_old = calcBitgroup(lastEB, levelSize, lossless_size);
            std::vector<std::vector<int>> bitGroupOfLayer_new = calcBitgroup(targetErrorBound, levelSize, lossless_size);

            // std::vector<std::vector<int>> bitGroupOfLayer_diff(layers, std::vector<int>(lsize, 0));
            

            return decompress(lossless_data, data, dec_data, bitGroupOfLayer_new, bitGroupOfLayer_old);
        }

        void loadcfg(uchar const *lossless_data, 
                    std::vector<size_t> &levelSize,
                    std::vector<size_t> &lossless_size, 
                    uchar const * &cmp_data_pos
                    ) {

            quantizer.postdecompress_data();

            uchar const *buffer = lossless_data;
            //load lossless_size
            size_t lossless_size_size = 0;
            read(lossless_size_size, buffer);
            lossless_size.clear();
            lossless_size.resize(lossless_size_size, 0);
            read(lossless_size.data(), lossless_size_size, buffer);

            //load dim && l2_diff
            size_t buffer_len = lossless_size[0];
            retrieved_size += buffer_len;
            buffer_len -= (lossless_size_size + 1) * sizeof(size_t);
            cmp_data_pos = lossless_data;
            read(global_dimensions.data(), N, buffer, buffer_len);
            num_elements = std::accumulate(global_dimensions.begin(), global_dimensions.end(), (size_t) 1, std::multiplies<>());
            read(interp_dim_limit, buffer, buffer_len);
            levelSize.clear();
            levelSize.resize(N * level_progressive, 0);
            read(levelSize.data(), levelSize.size(), buffer, buffer_len);
            {   // load unpredictable data
                {   // mv buffer pointer to the address of unpredictable data
                    buffer = lossless_data;         // ???
                    for (int i = 0; i < lossless_size.size() - 1; i++) {
                        buffer += lossless_size[i];
                    }
                    buffer_len = lossless_size[lossless_size.size() - 1];
                    retrieved_size += buffer_len;
                }
                size_t rSize = lossless.getFrameConteneSize(buffer, buffer_len);
                uchar * dcmpData = new uchar[rSize];
                size_t dcmpSize = lossless.decompress(buffer, buffer_len, dcmpData, rSize);
    //            uchar const *buffer_pos = buffer;
                uchar const * dcmpDataRef = dcmpData;
                quantizer.load(dcmpDataRef, dcmpSize);
                delete []dcmpData;
                // printf("[Log] Loading Unpredictable data...\n");
                // printf("[Log] retrieved = %.3f%% %lu\n", retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size);
            }
        }
        
        std::vector<std::vector<int>> calcBitgroup(T targetErrorBound, std::vector<size_t> const& levelSize, std::vector<size_t> const& size_lb){
            int lsize = N * level_progressive, bsize = bitgroup.size();
            std::vector<std::vector<int>> bitGroupOfLayer(layers, std::vector<int>(lsize, 0));
            if(targetErrorBound == 0){
                return bitGroupOfLayer;
            }
            assert(ebs.size() == bitGroupOfLayer.size());

            const int progressive_layer_limit = 4096;
            std::vector<int> throwawayBits(lsize, 0);

            auto valueTable = calcValueTable(size_lb);
            auto cost = calcUncerntaintyBitGroup();

            for(int i = 0; i < layers; i++){
                if(targetErrorBound >= ebs[i]){
                    // throwawayBits = strategy(levelSize, (int)floor(targetErrorBound / ebs[i]) - 1);
                    throwawayBits = strategy(valueTable, cost, (int)floor(targetErrorBound / ebs[i]) - 1);
                    for(int j = 0; j < lsize; j++){
                        // bitGroupOfLayer[i][j] =std::max(0, std::min(bsize - throwawayBits[j], bsize));
                        bitGroupOfLayer[i][j] = std::max(0, std::min(throwawayBits[j], bsize));
                    }
                    break;
                } else {
                    bitGroupOfLayer[i].assign(lsize, bsize);
                }
            }  
            return bitGroupOfLayer;
        }

        void *decompress(uchar const *lossless_data, T *data, T *dec_data,
                    std::vector<std::vector<int>> bitGroupOfLayer_new, 
                    std::vector<std::vector<int>> bitGroupOfLayer_old
                    ){
            int lsize = N * level_progressive, bsize = bitgroup.size();
            std::vector<int> bsum(lsize, 0), bdelta(lsize, bsize);
            uchar const * lossless_data_pos = lossless_data;

            std::vector<size_t> levelSize(lsize, 0);
            std::vector<size_t> lossless_size;
            uchar const * cmp_data_pos = nullptr;

            std::vector<std::vector<int>> bitGroupOfLayer_diff(layers, std::vector<int>(lsize, 0));
            for(int i = 0; i < layers; i++) {
                for(int j = 0; j < lsize; j++){
                    bitGroupOfLayer_diff[i][j] = bitGroupOfLayer_new[i][j] - bitGroupOfLayer_old[i][j];
                }
            }
            loadcfg(lossless_data_pos, levelSize, lossless_size, cmp_data_pos);

            size_t compressed_size = 0;
            if(!allzero(bitGroupOfLayer_diff[0])){
                
                bool update = !allzero(bitGroupOfLayer_old[0]);
                bsum = bitGroupOfLayer_old[0];
                bdelta = bitGroupOfLayer_diff[0];
                compressed_size = decompress(lossless_data_pos, dec_data, bsum, bdelta, levelSize, lossless_size, cmp_data_pos, update);
                {   // verification
                    double psnr, nrmse, max_err, range;
                    verify(data, dec_data, num_elements, psnr, nrmse, max_err, range);
                }
            } else {
                compressed_size = std::accumulate(lossless_size.begin(), lossless_size.end(), (size_t) 0);
                // std::cout << "[Log] skipping layer 0" << std::endl;
            }
            lossless_data_pos += compressed_size;
            {
                T *residual_data = new T[num_elements];
                for (int layer = 1; layer < ebs.size(); layer++){
                    
                    bsum.clear();
                    bsum.resize(lsize, 0);
                    
                    bool update = !allzero(bitGroupOfLayer_old[layer]);
                    bsum = bitGroupOfLayer_old[layer];

                    bdelta.clear();
                    bdelta.resize(lsize, bsize);
                    bdelta = bitGroupOfLayer_diff[layer];
                    loadcfg(lossless_data_pos, levelSize, lossless_size, cmp_data_pos);
                    if(allzero(bitGroupOfLayer_diff[layer])){ // bitGroupOfLayer uninitialized at this layer
                        compressed_size = std::accumulate(lossless_size.begin(), lossless_size.end(), (size_t) 0);
                        lossless_data_pos += compressed_size;
                        std::cout << "[log]skipping layer " << layer << std::endl;
                    } else {
                        compressed_size = decompress(lossless_data_pos, residual_data, bsum, bdelta, levelSize, lossless_size, cmp_data_pos, update);
                        lossless_data_pos += compressed_size;
                        // {
                        //     T a = dec_data[4823183];
                        //     T b = residual_data[4823183];
                        //     T c = data[4823183];
                        //     T z = a + b;
                        // }
                        for (size_t i = 0; i < num_elements; i++){
                            dec_data[i] += residual_data[i];
                        }
                        {   // verification
                            double psnr, nrmse, max_err, range;
                            verify(data, dec_data, num_elements, psnr, nrmse, max_err, range);
                        }
                    }
                    
                }
                delete []residual_data;
            }
            return dec_data;
        }

        size_t decompress(uchar const *lossless_data, 
                            T *dec_data,
                            std::vector<int>& bsum,
                            const std::vector<int>& bdelta,
                            std::vector<size_t> &levelSize,
                            std::vector<size_t> &lossless_size,
                            uchar const *cmp_data_pos,
                            bool update
                            ) {
            size_t compressed_size = std::accumulate(lossless_size.begin(), lossless_size.end(), (size_t) 0);
            quant_inds.reserve(num_elements);
            l2_diff.resize(level_progressive * N * bitgroup.size(), 0);
            // read(l2_diff.data(), l2_diff.size(), buffer, buffer_len);

            // size_t level_cnt = 0;

            Timer timer(true);
            //load non-progressive levels
            int lossless_id = 1;        // ???
            lossless_data += lossless_size[0];

            uchar const *prog_data_pos = lossless_data;

            // std::cout << "Progressive decompression..." << std::endl;
            int lsize = N * level_progressive, bsize = bitgroup.size();

            //load non-progressive levels
            std::vector<uchar const *> data_lb(lsize * bsize);
            std::vector<size_t> size_lb(lsize * bsize);
            //load all progressive levels into memory
            for (int l = 0; l < lsize; l++) {
                    for (int b = bsize - 1; b >= 0; b--) {
                        data_lb[l * bsize + b] = lossless_data;
                        size_lb[l * bsize + b] = lossless_size[lossless_id];
                        lossless_data += lossless_size[lossless_id];
                        lossless_id++;
                    }
            }

            if(level_progressive > 0)
            {
                decompress_progressive(dec_data,
                                    bsum, bdelta,
                                    bsize, lsize,
                                    data_lb, size_lb,
                                    levelSize, update);
            }
            quantizer.postdecompress_data();
            // printf("[Log] retrieved = %.3f%% %lu\n", retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size);

            return compressed_size;
            //decompress_progressive(dec_data, cmp_data_pos, prog_data_pos, lossless_size,
            //                    lossless_id, data, bsum, bdelta);
        }

        T *decompress_progressive(T* dec_data, 
                                std::vector<int>& bsum,
                                const std::vector<int>& bdelta,
                                const int bsize,
                                const int lsize,
                                std::vector<uchar const *> & data_lb,
                                std::vector<size_t> & size_lb,
                                std::vector<size_t> & levelSize,
                                // size_t & level_cnt,
                                bool update
                                ) {
            // size_t level_cnt_temp = level_cnt;
            {   // print eg.1 1 0 -> 1 1 1
                // printf("-----------------------\n");
                for (int l = 0; l < lsize; l++) {
                    printf("%d ", bsum[l]);
                }
                printf(" -> ");
                for (int l = 0; l < lsize; l++) {
                    printf("%d ", bsum[l] + bdelta[l]);
                }
                printf("\n");
            }
            Timer timer(true);
            timer.start();
            bool retrive = true;
            {   // retrive = if elements in bsum are all zeros
                for(auto i : bsum){
                    if(i){
                        retrive = false;
                        break;
                    }
                }
                // if(retrive && level_progressive != levels){
                //     for (uint l = level_progressive; l > 0; l--) {
                //         for (const auto &direction: directions) {
                //             block_interpolation(dec_data, dec_data, global_begin, global_end, &SZProgressiveMQuant::recover_no_quant,
                //                                 interpolators[interpolator_id], direction, 1U << (l - 1), true);
                //         }
                //     }
                // }
            }

            ska::unordered_map<std::string, double> result;
            dec_delta.clear();
            dec_delta.resize(num_elements, 0);
            for (uint level = level_progressive; level > 0; level--) {
                for (int direct = 0; direct < N; direct++) {
                    int lid = (level_progressive - level) * N + direct;

                    result["level"] = level;
                    result["direct"] = direct;
                    size_t quant_size = levelSize[(level_progressive - level) * N + direct];
                    int bg_end = std::min(bsize, bsum[lid] + bdelta[lid]);
                    {   // load bit group data into quant_ids[ ]
                        quant_inds.clear();
                        quant_cnt = 0;
                        quant_inds.resize(quant_size, 0);
                        if (bdelta[lid] > 0){
                            for (int b = bsum[lid]; b < bg_end; b++) {
                                // l2_proj = l2_diff[lid * bsize + b];
//                                    printf("projected l2 delta = %.10G\n", l2_diff[lid * bsize + b]);
                                uchar const *bg_data = data_lb[lid * bsize + b];
                                size_t bg_len = size_lb[lid * bsize + b];
                                lossless_decode_bitgroup(b, bg_data, bg_len, quant_size);
                                // printf("--------[Log] bitGroup_len = %d\n", bg_len);
                            }
                        }
                        // level_cnt++; 
                    }
                    if(level_progressive == levels && lid == 0) // retrive
                    {
                        if(retrive){
                            *dec_data = quantizer.recover(0, 0, quant_inds[quant_cnt++]);
                            for (uint l = level_progressive; l > 0; l--) {
                                for (const auto &direction: directions) {
                                    block_interpolation(dec_data, dec_data, global_begin, global_end, &SZProgressiveMQuant::recover_no_quant,
                                                        interpolators[interpolator_id], direction, 1U << (l - 1), true);
                                }
                            }
                        } else {
                            quant_cnt++;
                        }
                    }
                    if (!update) {
                        block_interpolation(dec_data, dec_data, global_begin, global_end, &SZProgressiveMQuant::recover,
                                            interpolators[interpolator_id], directions[direct], 1U << (level - 1), true);
                    } else {
                        block_interpolation(dec_delta.data(), dec_delta.data(), global_begin, global_end,
                                            &SZProgressiveMQuant::recover_only_set_delta,
                                            interpolators[interpolator_id], directions[direct], 1U << (level - 1), true);
                    }
                    bsum[lid] = bg_end;
                }
            }  
            if (update) {
                for (size_t idx = 0; idx < num_elements; idx++) {
                    quantizer.recover_from_delta(idx, dec_data[idx], dec_delta[idx]);
                }
            }
            // level_cnt = level_cnt_temp;
            // {   // verification
            //     double psnr, nrmse, max_err, range, l2_no_propo = 0;
            //     verify(data, dec_data, num_elements, psnr, nrmse, max_err, range, l2_no_propo);
            //     printf("------[Log] retrieved = %.3f%% %lu\n", retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size);
            // }
            std::cout << "decompress time = " << timer.stop() << std::endl;
            return dec_data;
        }

        uchar *compress(T *data, size_t & compressed_size) {
            setupLayers(data);
            uchar *lossless_data = new uchar[size_t((num_elements < 1000000 ? 100 : 2.0) * num_elements) * sizeof(T)]; //?
            uchar * lossless_data_pos = lossless_data;
            
            
            // std::vector<T> err_data{error};
            // uchar *lossless_data_new = new uchar[size_t((num_elements < 1000000 ? 100 : 1.5) * num_elements) * sizeof(T)]; //?
            bool isFirst = true;
            for(auto eb : ebs){
                quantizer.set_eb(eb);
                if(isFirst){
                    isFirst = false;
                    compressed_size = compress(data, lossless_data_pos);
                    lossless_data_pos += compressed_size;
                } else {
                    memcpy(data, error.data(), num_elements * sizeof(T));
                    size_t delta_compressed_data = compress(data, lossless_data_pos);
                    compressed_size += delta_compressed_data;
                    lossless_data_pos += delta_compressed_data;
                }
            }
            return lossless_data;
        }
        
        // compress given the error bound
        size_t compress(T *data, uchar* lossless_data) {

            // {
            //     std::vector<size_t> levelSize{1, 5, 10};
            //     strategy(levelSize, 10);

            //     int qu = ((int32_t) 7 + (uint32_t) 0xaaaaaaaau) ^ (uint32_t) 0xaaaaaaaau;
            //     int dequ = (((uint32_t) qu) ^ 0xaaaaaaaau) - 0xaaaaaaaau;
            //     std::cout << dequ << std::endl;

            // }


            Timer timer(true);

            quant_inds.reserve(num_elements);
            error.resize(num_elements, 0);
            size_t interp_compressed_size = 0;
            size_t quant_inds_total = 0;

            

            T eb = quantizer.get_eb();
            std::cout << "Absolute error bound = " << eb << std::endl;
//            quantizer.set_eb(eb * eb_ratio);
            l2_diff.resize(level_progressive * N * bitgroup.size(), 0);

            uchar *lossless_data_pos = lossless_data;

            // save space for lossless_size;
            size_t estimated_lossless_size_size = 1 + N * level_progressive * bitgroup.size() + 1;
            std::vector<size_t> lossless_size;
            lossless_size.reserve(estimated_lossless_size_size);
            write(estimated_lossless_size_size, lossless_data_pos);
            uchar *lossless_size_pos = lossless_data_pos;
            lossless_data_pos += estimated_lossless_size_size * sizeof(size_t);

            write(global_dimensions.data(), N, lossless_data_pos);
            write(interp_dim_limit, lossless_data_pos);

            // element # of each level
            uchar *levelSize_pos = lossless_data_pos;
            memset(levelSize_pos, 0, levels * N * sizeof(size_t));
            lossless_data_pos += levels * N * sizeof(size_t);

            uchar *error_mse_pos = lossless_data_pos;
           
            // lossless_data_pos += l2_diff.size() * sizeof(T);
            lossless_size.push_back(lossless_data_pos - lossless_data);

            timer.start();
            
            for (uint level = level_progressive; level > 0; level--) {

//                quantizer.set_eb((level >= 3) ? eb * eb_ratio : eb);
                uint stride = 1U << (level - 1);
                if (level == levels) {
                    // quant_inds.push_back(quantizer.quantize_and_overwrite(0, *data, 0));
                    quantize(0, data[0], 0);
                }
                for (int d = 0; d < N; d++) {
                    block_interpolation(data, data, global_begin, global_end, &SZProgressiveMQuant::quantize,
                                        interpolators[interpolator_id], directions[d], stride, true);

                    auto quant_size = quant_inds.size();
                    quant_inds_total += quant_size;
                    write(quant_size, levelSize_pos);
                    auto size = encode_lossless_bitplane((level_progressive - level) * N + d, lossless_data_pos, lossless_size, eb);
                    // printf("level = %d , direction = %d, quant size = %lu, lossless size = %lu, time=%.3f\n\n",
                    //        level, d, quant_size, size, timer.stop());

                }
            }

//            quant_inds.clear();
            std::cout << "total element = " << num_elements << ", quantization element = " << quant_inds_total << std::endl;
            std::cout << "compress time = " << timer.stop() << std::endl;
            assert(quant_inds_total >= num_elements);

            // write(l2_diff.data(), l2_diff.size(), error_mse_pos);

            uchar *buffer = new uchar[quantizer.get_unpred_size() * (sizeof(T) + sizeof(size_t)) + 40];
            uchar *buffer_pos = buffer;
            quantizer.save(buffer_pos);
            size_t size = lossless.compress(buffer, buffer_pos - buffer, lossless_data_pos);
            delete[] buffer;
            
            lossless_data_pos += size;
            lossless_size.push_back(size);

            assert(lossless_size.size() == estimated_lossless_size_size);
            write(lossless_size.data(), lossless_size.size(), lossless_size_pos);
            // ----- 
            // quantizer.set_eb(1e-5);
            // compress_progressive(error.data(), lossless_size, lossless_data);
            quantizer.postcompress_data();
            return std::accumulate(lossless_size.begin(), lossless_size.end(), (size_t) 0);

        }

    private:
        typedef void (SZProgressiveMQuant::*PredictionFunc)(size_t, T &, T);

        int levels = -1;
        int level_progressive = -1;
        int layers = -1;
//        int level_fill = 0;
        int interpolator_id;
        size_t interp_dim_limit, block_size;
        double eb_ratio = 0.5;
        T range = 0;
        std::vector<T> ebs;
        std::vector<std::string> interpolators;
        std::vector<int> quant_inds;
        std::vector<T> error;
        std::vector<T> l2_diff;
        size_t quant_cnt = 0; // for decompress
        size_t num_elements;
        std::array<size_t, N> global_dimensions, global_begin, global_end;
        std::array<size_t, N> dim_offsets;
        std::array<std::pair<std::array<int, N>, std::array<int, N - 1>>, N> directions;
        Quantizer quantizer;
        Encoder encoder;
        Lossless lossless;


    //    std::vector<int> bitgroup = {8, 8, 8, 2, 2, 2, 1, 1};
//TODO quantizati45on bins in different levels have different distribution.
// a dynamic bitgroup should be used for each level
       std::vector<int> bitgroup = {16, 8, 4, 2, 1, 1};
        // std::vector<int> bitgroup = {16, 8, 2, 2, 1, 1, 1, 1};
    //    std::vector<int> bitgroup = {4, 4, 4, 4, 4, 4, 4, 4,};
    //    std::vector<int> bitgroup = {16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    //    std::vector<int> bitgroup = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<T> dec_delta;
        size_t retrieved_size = 0;

        //debug only
        double max_error;
//        T eb;
        void
        lossless_decode_bitgroup(int bg, uchar const *data_pos, const size_t data_length, const size_t quant_size) {
            Timer timer(true);

            size_t length = data_length;
            retrieved_size += length;
            if (length == 0) {return; }
            // artimetic
            // ---------------
            
            // zstd
            // size_t rSize = lossless.getFrameConteneSize(data_pos, length);
            // uchar *compressed_data = new uchar[rSize];
            // length = lossless.decompress(data_pos, length, compressed_data, rSize);

            //
            // uchar *compressed_data = new uchar[length];
            // memcpy(compressed_data, data_pos, length);
            // std::vector<int> quant_ind_truncated;
            // uchar const *compressed_data_pos = compressed_data;
            // encoder.load(compressed_data_pos, length);
            // quant_ind_truncated = encoder.decode(compressed_data_pos, quant_size);
            // encoder.postprocess_decode();


            // artimetic ends
            // ---------------
            
            // huffman && zstd
            // ---------------
            uchar * compressed_data = nullptr;
            if (quant_size < 128 && bitgroup[bg] == 1) {
                compressed_data = new uchar[length];
                memcpy(compressed_data, data_pos, length);
            } else {
                size_t rSize = lossless.getFrameConteneSize(data_pos, length);
                compressed_data = new uchar[rSize];
                length = lossless.decompress(data_pos, length, compressed_data, rSize);
            }
            uchar const *compressed_data_pos = compressed_data;

            // size_t quant_size;
            // read(quant_size, compressed_data_pos, length);

            std::vector<int> quant_ind_truncated;
            if (bitgroup[bg] == 1) {
                quant_ind_truncated = decode_int_1bit(compressed_data_pos, length, quant_size);
            } else if (bitgroup[bg] == 2) {
                quant_ind_truncated = decode_int_2bits(compressed_data_pos, length);
            } else {
                encoder.load(compressed_data_pos, length);
                quant_ind_truncated = encoder.decode(compressed_data_pos, quant_size);
                encoder.postprocess_decode();
            }

            // lossless.postdecompress_data(compressed_data);
            // huffman && zstd ends
            // ---------------
            delete[] compressed_data;


//                printf("\n************Bitplane = %d *****************\n", bg);
            int bitshift = 32;
            for (int bb = 0; bb <= bg; bb++) {
                bitshift -= bitgroup[bb];
            }
            // std::cout << "------[Log] quant size = " << quant_size << std::endl;
            for (size_t i = 0; i < quant_size; i++) {
                quant_inds[i] += (((uint32_t) quant_ind_truncated[i] << bitshift) ^ 0xaaaaaaaau) - 0xaaaaaaaau;
                // quant_inds[i] += (((uint32_t) quant_ind_truncated[i] << bitshift)) - 0xaaaaaaaau;

                // int bitshiftcnt = bitshift;
                // if(bitshiftcnt < 16){
                //     for (int j = 0; j < bitgroup[bg]; j++){
                //         int quant_ind_truncated_bit = (quant_ind_truncated[i] >> j) & 1;
                //         int add = (bitshiftcnt == 0) ? quant_ind_truncated_bit : ((quant_ind_truncated_bit == 1 ) ? (1 << (bitshiftcnt - 1)) : - (1 << (bitshiftcnt - 1)));
                //         quant_inds[i] += add;
                //         bitshiftcnt++;
                //     }
                // }
                // if(bitshift < 16) {
                //     // int offset = ((((((unsigned int)(1 << 15) - 1) >> bitshift) << bitshift) << (bitshift + bitgroup[bg])) >> (bitshift + bitgroup[bg]));
                //     int offset = (1 << 15) - 1;
                //     quant_inds[i] += (quant_ind_truncated[i] << bitshift) - offset;
                // }

            }
            // std::cout << "------[Log] quant_size = " << quant_size << std::endl;
            // std::cout << "------[Log] bg = " << bg << std::endl;
        }


        size_t encode_lossless_bitplane(int lid, uchar *&lossless_data_pos, std::vector<size_t> &lossless_size, T eb) {
            Timer timer;
            int bsize = bitgroup.size();
            size_t qsize = quant_inds.size();
            std::vector<int> quants(qsize);

            uchar *buffer = new uchar[size_t((quant_inds.size() < 1000000 ? 10 : 1.2)
                                             * quant_inds.size()) * sizeof(T)];
            {   // convert quant_inds to negabinary based
                for (size_t i = 0; i < qsize; i++) {
                    quant_inds[i] = ((int32_t) quant_inds[i] + (uint32_t) 0xaaaaaaaau) ^ (uint32_t) 0xaaaaaaaau;
                    // quant_inds[i] = ((int32_t) quant_inds[i] + (uint32_t) 0xaaaaaaaau);

                    
                    // quant_inds[i] += (1 << 15) - 1;
                }
            }

            double l2_error_base = 0;
            {   // calc the total l2 error
                for (size_t i = 0; i < qsize; i++) {
                    if(i < error.size()) {
                        // l2_error_base += error[i] * error[i];
                    }
                }
                // printf("l2 = %.10G \n", l2_error_base);
            }
            size_t total_size = 0;
            int shift = 0;
            for (int b = bsize - 1; b >= 0; b--) {
                timer.start();
                uchar *buffer_pos = buffer;
                // write((size_t) qsize, buffer_pos);

                double l2_error = 0;
                for (size_t i = 0; i < qsize; i++) {
                    quants[i] = quant_inds[i] & (((uint64_t) 1 << bitgroup[b]) - 1);
                    quant_inds[i] >>= bitgroup[b];
                    int qu = (((uint32_t) quants[i] << shift) ^ 0xaaaaaaaau) - 0xaaaaaaaau;
                    if (i < error.size()) {
                        // error[i] += qu * 2.0 * eb;
                        // l2_error += error[i] * error[i];
                    }
                }
                l2_diff[lid * bsize + b] = l2_error - ((b == bsize - 1) ? l2_error_base : l2_diff[lid * bsize + b + 1]);
                // printf("l2 = %.10G , diff = %.10G\n", l2_error, l2_diff[lid * bsize + b]);
                shift += bitgroup[b];


                // ---------------
                // arithemetic
                // encoder.preprocess_encode(quants, 2);
                // encoder.save(buffer_pos);
                // encoder.encode(quants, buffer_pos);
                // encoder.postprocess_encode();
                
                // memcpy(lossless_data_pos, buffer, buffer_pos - buffer);
                // size_t size = buffer_pos - buffer;
                // total_size += size;
                // lossless_data_pos += size;
                // lossless_size.push_back(size);
                // zstd
                // size_t size = lossless.compress(
                //             buffer, buffer_pos - buffer, lossless_data_pos);
                // total_size += size;
                // lossless_data_pos += size;
                // lossless_size.push_back(size);

                // arithemetic end
                // ---------------

                // ---------------
                // huffman && zstd
                if(quants.size() > 0){
                    if (bitgroup[b] == 1) {
                        encode_int_1bit(quants, buffer_pos);
                    } else if (bitgroup[b] == 2) {
                        encode_int_2bits(quants, buffer_pos);
                    } else {
                        //TODO huffman tree is huge if using large radius on early levels
                        // set different radius for each level
                        encoder.preprocess_encode(quants, 0);
                        encoder.save(buffer_pos);
                        encoder.encode(quants, buffer_pos);
                        encoder.postprocess_encode();
                    }
                    if(qsize < 128 && bitgroup[b] == 1) {
                        memcpy(lossless_data_pos, buffer, buffer_pos - buffer);
                        size_t size = buffer_pos - buffer;
        //                printf("%d %lu, ", bitgroup[b], size);
                        total_size += size;
                        lossless_data_pos += size;
                        lossless_size.push_back(size);
                    } else {
                        size_t size = lossless.compress(
                                buffer, buffer_pos - buffer, lossless_data_pos);
        //                printf("%d %lu, ", bitgroup[b], size);
                        total_size += size;
                        lossless_data_pos += size;
                        lossless_size.push_back(size);
                    }
                } else {
                    lossless_size.push_back(0);
                }


                // huffman && zstd ends
                // ---------------
            }
//            printf("\n");
            
            delete[]buffer;
            quant_inds.clear();
            // error.clear();

            return total_size;
        }

        void lossless_decode(uchar const *&lossless_data_pos, const std::vector<size_t> &lossless_size, int lossless_id, size_t quant_size) {

            // size_t remaining_length = lossless_size[lossless_id];
            retrieved_size += lossless_size[lossless_id];

            size_t rSize = lossless.getFrameConteneSize(lossless_data_pos, lossless_size[lossless_id]);
            uchar *compressed_data = new uchar[rSize];
            size_t dcmpSize = lossless.decompress(lossless_data_pos, lossless_size[lossless_id], compressed_data, rSize);
            uchar const *compressed_data_pos = compressed_data;

            // size_t quant_size;
            // read(quant_size, compressed_data_pos, remaining_length);
            //                printf("%lu\n", quant_size);
            if (quant_size < 128) {
                quant_inds.resize(quant_size);
                read(quant_inds.data(), quant_size, compressed_data_pos, dcmpSize);
            } else {
                encoder.load(compressed_data_pos, dcmpSize);
                quant_inds = encoder.decode(compressed_data_pos, quant_size);
                encoder.postprocess_decode();
            }
            quant_cnt = 0;

            // lossless.postdecompress_data(compressed_data);
            delete []compressed_data;
            lossless_data_pos += lossless_size[lossless_id];
        }

        size_t encode_lossless(uchar *&lossless_data_pos, std::vector<size_t> &lossless_size) {
            uchar *compressed_data = new uchar[size_t((quant_inds.size() < 1000000 ? 10 : 1.2) * quant_inds.size()) * sizeof(T)];
            uchar *compressed_data_pos = compressed_data;

            // write((size_t) quant_inds.size(), compressed_data_pos);
            if (quant_inds.size() < 128) {
                write(quant_inds.data(), quant_inds.size(), compressed_data_pos);
            } else {
                // encoder.preprocess_encode(quant_inds, 0);// for huffman
                encoder.preprocess_encode(quant_inds, 0);// for huffman
                encoder.save(compressed_data_pos);
                encoder.encode(quant_inds, compressed_data_pos);
                encoder.postprocess_encode();
            }

            size_t size = lossless.compress(compressed_data, compressed_data_pos - compressed_data,
                                            lossless_data_pos);
            // lossless.postcompress_data(compressed_data);
            // {
            //     remaining_length = lossless.decompress(lossless_data_pos, remaining_length, compressed_data, num_elements * sizeof(T));
            //     uchar const *compressed_data_pos = compressed_data;
            // }
            delete []compressed_data;
            lossless_data_pos += size;
            lossless_size.push_back(size);

            quant_inds.clear();
            // error.clear();

            
            return size;
        }

        inline void quantize(size_t idx, T &data, T pred) {
            T data0 = data;
            quant_inds.push_back(quantizer.quantize_and_overwrite(idx, data, pred));
            // error.push_back(data0 - data);
            error[idx] = data0 - data;

        }

        inline void recover(size_t idx, T &d, T pred) {
            d = quantizer.recover(idx, pred, quant_inds[quant_cnt++]);
        };

        inline void recover_only_quant(size_t idx, T &d, T pred) {
            d = quantizer.recover(idx, 0, quant_inds[quant_cnt++]);
        };

        inline void recover_no_quant(size_t idx, T &d, T pred) {
            d = quantizer.recover(idx, pred, 0);
        };

        inline void recover_set_delta_no_quant(size_t idx, T &d, T pred) {
            quantizer.recover_and_residual(idx, d, dec_delta[idx], pred);
        };

        inline void recover_set_delta(size_t idx, T &d, T pred) {
            quantizer.recover_and_residual(idx, d, dec_delta[idx], pred, quant_inds[quant_cnt++]);
        };

        inline void recover_only_set_delta(size_t idx, T&d, T pred) {
            quantizer.recover_only_set_delta(idx, dec_delta[idx], pred, quant_inds[quant_cnt++]);
        }


        inline void fill(size_t idx, T &d, T pred) {
            d = pred;
        }


        double block_interpolation_1d(T *d, T *pd, size_t begin, size_t end, size_t stride,
                                      const std::string &interp_func, PredictionFunc func) {
            size_t n = (end - begin) / stride + 1;
            if (n <= 1) {
                return 0;
            }

            size_t c;
            size_t stride3x = stride * 3, stride5x = stride * 5;
            if (interp_func == "linear" || n < 5) {
                size_t i = 1;
                for (i = 1; i + 1 < n; i += 2) {
                    c = begin + i * stride;
                    (this->*func)(c, d[c], interp_linear(pd[c - stride], pd[c + stride]));
                }
                if (n % 2 == 0) {
                    c = begin + (n - 1) * stride;
                    if (n < 4) {
                        (this->*func)(c, d[c], pd[c - stride]);
                    } else {
                        (this->*func)(c, d[c], interp_linear1(pd[c - stride3x], pd[c - stride]));
                    }
                }
            } else {
                size_t i = 1;
                c = begin + i * stride;
                (this->*func)(c, d[c], interp_quad_1(pd[c - stride], pd[c + stride], pd[c + stride3x]));
                for (i = 3; i + 3 < n; i += 2) {
                    c = begin + i * stride;
                    (this->*func)(c, d[c], interp_cubic(pd[c - stride3x], pd[c - stride], pd[c + stride], pd[c + stride3x]));
                }
                c = begin + i * stride;
                (this->*func)(c, d[c], interp_quad_2(pd[c - stride3x], pd[c - stride], pd[c + stride]));
                if (n % 2 == 0) {
                    c = begin + (n - 1) * stride;
                    (this->*func)(c, d[c], interp_quad_3(pd[c - stride5x], pd[c - stride3x], pd[c - stride]));
                }
            }
            return 0;
        }


        void block_interpolation(T *data, T *pred_data, std::array<size_t, N> begin, std::array<size_t, N> end, PredictionFunc func,
                                 const std::string &interp_func, const std::pair<std::array<int, N>, std::array<int, N - 1>> direction,
                                 uint stride, bool overlap) {

            auto dims = direction.first;
            auto s = direction.second;

            if (N == 1) {
                block_interpolation_1d(data, pred_data, begin[0], end[0], stride, interp_func, func);
            } else if (N == 2) {
                for (size_t i = begin[dims[0]] + ((overlap && begin[dims[0]]) ? stride * s[0] : 0); i <= end[dims[0]]; i += stride * s[0]) {
                    size_t begin_offset = i * dim_offsets[dims[0]] + begin[dims[1]] * dim_offsets[dims[1]];
                    block_interpolation_1d(data, pred_data, begin_offset, begin_offset + (end[dims[1]] - begin[dims[1]]) * dim_offsets[dims[1]],
                                           stride * dim_offsets[dims[1]], interp_func, func);
                }
            } else if (N == 3) {
                for (size_t i = begin[dims[0]] + ((overlap && begin[dims[0]]) ? stride * s[0] : 0); i <= end[dims[0]]; i += stride * s[0]) {
                    for (size_t j = begin[dims[1]] + ((overlap && begin[dims[1]]) ? stride * s[1] : 0); j <= end[dims[1]]; j += stride * s[1]) {
                        size_t begin_offset = i * dim_offsets[dims[0]] + j * dim_offsets[dims[1]] + begin[dims[2]] * dim_offsets[dims[2]];
                        block_interpolation_1d(data, pred_data, begin_offset, begin_offset + (end[dims[2]] - begin[dims[2]]) * dim_offsets[dims[2]],
                                               stride * dim_offsets[dims[2]], interp_func, func);
                    }
                }
            } else {
                for (size_t i = begin[dims[0]] + ((overlap && begin[dims[0]]) ? stride * s[0] : 0); i <= end[dims[0]]; i += stride * s[0]) {
                    for (size_t j = begin[dims[1]] + ((overlap && begin[dims[1]]) ? stride * s[1] : 0); j <= end[dims[1]]; j += stride * s[1]) {
                        for (size_t k = begin[dims[2]] + ((overlap && begin[dims[2]]) ? stride * s[2] : 0);
                             k <= end[dims[2]]; k += stride * s[2]) {
                            size_t begin_offset = i * dim_offsets[dims[0]] + j * dim_offsets[dims[1]] + k * dim_offsets[dims[2]] +
                                                  begin[dims[3]] * dim_offsets[dims[3]];
                            block_interpolation_1d(data, pred_data, begin_offset,
                                                   begin_offset + (end[dims[3]] - begin[dims[3]]) * dim_offsets[dims[3]],
                                                   stride * dim_offsets[dims[3]], interp_func, func);
                        }
                    }
                }
            }
        }

        void set_directions_and_stride(int direction_id_) {
            std::array<int, N> base_direction;
            std::array<int, N - 1> stride_multiplication;
            for (int i = 0; i < N - 1; i++) {
                base_direction[i] = i;
                stride_multiplication[i] = 2;
            }
            base_direction[N - 1] = N - 1;

            int direction_id = 0;
            do {
                if (direction_id_ == direction_id) {
                    for (int i = 0; i < N - 1; i++) {
                        auto direction = base_direction;
                        std::rotate(direction.begin() + i, direction.begin() + i + 1, direction.end());
                        directions[i] = std::pair{direction, stride_multiplication};
                        stride_multiplication[i] = 1;
                    }
                    directions[N - 1] = std::pair{base_direction, stride_multiplication};
                    break;
                }
                direction_id++;
            } while (std::next_permutation(base_direction.begin(), base_direction.end()));

            for (int i = 0; i < N; i++) {
                printf("direction %d is ", i);
                for (int j = 0; j < N; j++) {
                    printf("%d ", directions[i].first[j]);
                }
                printf("\n");
            }
        }

        void global_index(size_t offset) {
            std::array<size_t, N> global_idx{0};
            for (int i = N - 1; i >= 0; i--) {
                global_idx[i] = offset % global_dimensions[i];
                offset /= global_dimensions[i];
            }
            for (const auto &id: global_idx) {
                printf("%lu ", id);
            }
        }

        void getRange(T* data) {
            T max = data[0];
            T min = data[0];
            for (size_t i = 1; i < num_elements; i++) {
                if (max < data[i]) max = data[i];
                if (min > data[i]) min = data[i];
            }
            range = max - min;
        }

        unsigned int calcUncerntaintyBalance(unsigned int bit) {
            assert(bit < 32);
            return (bit == 0) ? 0 : (1 << bit);
        }

        unsigned int calcUncerntaintyNegaBinary(unsigned int bit) {
            assert(bit < 32);
            return (bit == 0) ? 0 : (0xaaaaaaaau >> (32 - bit)) << 1;
        }

        // std::vector<unsigned int> calcUncerntaintyBitGroup() {
        //     std::vector<unsigned int> uncerntainty, throwbits;
        //     int accum = 0;
        //     for(auto bg : bitgroup){
        //         throwbits.push_back(32 - accum); //quantized 32 bits integer
        //         accum += bg;
        //     }

        //     return uncerntainty;
        // }
        std::vector<std::vector<int>> calcUncerntaintyBitGroup() {
            std::vector<unsigned int> throwbits;
            std::vector<std::vector<int>> uncerntainty(level_progressive * N, std::vector<int>(bitgroup.size() + 1, 0));
            int accum = 0;
            for(auto bg : bitgroup){
                throwbits.push_back(32 - accum); //quantized 32 bits integer
                accum += bg;
            }
            throwbits.push_back(32 - accum);
            #include<cmath>
            for(int i = 0; i < level_progressive * N; i++) {
                for (int b = 0; b < throwbits.size(); b++) {
                    auto bitshift = throwbits[b];
                    double cost = 0;
                    if(bitshift == 32) {
                        cost = std::pow(2, 32);
                    } else {
                        // cost = calcUncerntaintyBalance(bitshift);
                        cost = calcUncerntaintyNegaBinary(bitshift);
                    }
                    if(interpolators[interpolator_id] == "cubic") {
                        cost *= std::pow(1.25, level_progressive * N - 1 - i);
                    }
                    uncerntainty[i][b] = (int)ceil(cost);
                }
            }
            return uncerntainty;
        }
        std::vector<std::vector<size_t>> calcValueTable(std::vector<size_t> const& size_lb){
            assert(size_lb.size() % bitgroup.size() == 0);
            int levels = size_lb.size() / bitgroup.size();
            std::vector<std::vector<size_t>> valueTable(levels, std::vector<size_t>(bitgroup.size() + 1, 0));
            for(int l = 0; l < levels; l++) {
                for(int b = bitgroup.size() - 1; b >= 0; b--) {
                    valueTable[l][b] = valueTable[l][b + 1] + size_lb[(l + 1) * bitgroup.size() - b - 1];
                }
            }
            return valueTable;
        }

        std::vector<int> strategy(std::vector<std::vector<size_t>> const& valueTable, std::vector<std::vector<int>> const& cost, unsigned long long int limit) {
            // assert(valueTable.size() % cost.size() == 0);

            int levels = valueTable.size();
            int bgSize = cost[0].size(); // cost.size = bitgroup.size + 1
            std::vector<std::vector<long long>> dp(levels + 1, std::vector<long long>(limit + 1, 0));
            for (size_t i = 1; i <= levels; i++){
                for(size_t j = 0; j <= limit; j++){
                    long long maxValue = 0;
                    size_t max_j = 0;
                    for(int k = 0; k < bgSize; k++){// k : index of the threw away bit group
                        unsigned int cost_k = cost[i - 1][k];

                        // unsigned int cost_k = (k == 0) ? 0 : (1 << k);

                        if (j >= cost_k){
                            long long value = dp[i - 1][j - cost_k] + valueTable[i - 1][k];
                            if (value > maxValue){
                                max_j = j;
                                maxValue = value;
                            }
                        }
                    }
                    dp[i][j] = maxValue;
                }
            }
            double remaining_limit = limit;
            std::vector<int> keptBitGroup(levels, 0);
            for (size_t i = levels; i >= 1; i--){
                for (int k = 0; k < bgSize; k++){
                    unsigned int cost_k =  cost[i - 1][k];
                    // unsigned int cost_k = (k == 0) ? 0 : (1 << k);

                    if (remaining_limit < cost_k) {continue;}
                    if (dp[i][remaining_limit] == dp[i - 1][remaining_limit - cost_k] + valueTable[i - 1][k]){
                        // keptBitGroup[i - 1] = (k == bgSize - 1) ? -1 : k;
                        keptBitGroup[i - 1] = k;

                        remaining_limit -= cost_k;
                        break;
                    }
                }
            }
            return keptBitGroup;
        }

        std::vector<int> strategy(const std::vector<size_t> &levelSize, int limit) {
            #include<cmath>
            std::vector<size_t> levelSizeTemp{levelSize};
            if(interpolators[interpolator_id] == "cubic") {
                int sz = levelSize.size();
                for(int i = 0; i < sz; i++){
                    levelSizeTemp[i] = (size_t) (ceil(std::pow(1.25, (sz - 1 - i)) * levelSizeTemp[i]));
                }
            }
            // assert(levelSize.size() == N * level_progressive);
            assert(limit >= 0);
            std::vector<int> throwawayBits;
            size_t sizelb = levelSize.size();
            throwawayBits.resize(sizelb, 0);

            int BITGROUP_SEARCHING_LIMIT = 30;
            std::vector<int> NonZeroMap(sizelb, -1);
            size_t NewSizelb = sizelb;
            for(size_t i = 0; i < sizelb; i++){
                bool foundOne = false;
                for(size_t j = (i > 0) ? NonZeroMap[i - 1] + 1 : 0; j < sizelb; j++){
                    if(levelSizeTemp[j] > 0){
                        NonZeroMap[i] = j;
                        foundOne = true;
                        break;
                    }
                }
                if (!foundOne){
                    NewSizelb = i;
                    break;
                }
            }
            std::vector<size_t> NonZeroLevelSize;
            NonZeroLevelSize.reserve(NewSizelb);
            for(size_t i = 0; i < NewSizelb; i++){
                if(NonZeroMap[i] >= 0){
                    NonZeroLevelSize.push_back(levelSizeTemp[NonZeroMap[i]]);
                } else {
                    break;
                }
            }

            std::vector<std::vector<long long>> dp(NewSizelb + 1, std::vector<long long>(limit + 1, 0));

            for (size_t i = 1; i <= NewSizelb; i++){
                for(size_t j = 0; j <= limit; j++){
                    long long maxValue = 0;
                    size_t max_j = 0;
                    for(int k = 0; k <= BITGROUP_SEARCHING_LIMIT; k++){
                        unsigned int cost_k = (k == 0) ? 0 : (0xaaaaaaaau >> (32 - k)) << 1;
                        // unsigned int cost_k = (k == 0) ? 0 : (1 << k);

                        if (j >= cost_k){
                            long long value = dp[i - 1][j - cost_k] + k * NonZeroLevelSize[i - 1];
                            if (value > maxValue){
                                max_j = j;
                                maxValue = value;
                            }
                        }
                    }
                    dp[i][j] = maxValue;
                }
            }
            size_t remaining_limit = limit;
            for (size_t i = NewSizelb; i >= 1; i--){
                for (int k = 0; k <= BITGROUP_SEARCHING_LIMIT; k++){
                    unsigned int cost_k = (k == 0) ? 0 : (0xaaaaaaaau >> (32 - k)) << 1;
                    // unsigned int cost_k = (k == 0) ? 0 : (1 << k);


                    if (remaining_limit < cost_k) {break;}
                    if (dp[i][remaining_limit] == dp[i - 1][remaining_limit - cost_k] + k * NonZeroLevelSize[i - 1]){
                        throwawayBits[NonZeroMap[i - 1]] = k;
                        remaining_limit -= cost_k;
                        break;
                    }
                }
            }
            return throwawayBits;
        }

        void init(){
            quant_cnt = 0;
            quant_inds.clear();
            error.clear();
        }

        bool allzero(std::vector<int> &a) {
            for(auto ele : a) {
                if(ele) {
                    return false;
                }
            }
            return true;
        }

        void setupLayers(T *data){
            getRange(data);
            printf("Value Range = %.4f\n", range);
            switch (layers)
            {
            case 1:
                ebs = {(T)(range * 1e-6)};
                // ebs = {(T)(1e-6)};

                break;
            case 2:
                ebs = {(T)(range * 1e-3), (T)(range * 1e-6)};
                // ebs = {(T)(1e-3), (T)(1e-6)};
                break;
            case 3:
                ebs = {(T)(range * 1e-2), (T)(range * 1e-4), (T)(range * 1e-6)};
                // ebs = {(T)(1e-2), (T)(1e-4), (T)(1e-6)};
                break;
            default:
                ebs = {(T)(range * 1e-6)};
                // ebs = {(T)(1e-6)};
                std::cout << "[warning] param 'layers' too large." << std::endl;
                layers = 1;
                break;
            }
            layers = ebs.size();
        }
    };
};


#endif

