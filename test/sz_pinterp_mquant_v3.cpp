//#include <compressor/SZProgressiveIndependentBlock.hpp>
//#include <compressor/SZProgressive.hpp>
#include <SZ3/compressor/SZProgressiveMQuantV3.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>
#include <SZ3/predictor/ComposedPredictor.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/encoder/ArithmeticEncoder.hpp>
#include <SZ3/utils/Iterator.hpp>
#include <SZ3/utils/Verification.hpp>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <memory>
#include <type_traits>
#include <sstream>

template<uint N, typename T, class ... Dims>
SZ3::uchar *interp_compress(const char *path, int interp_op, int direction_op,
                                int layers, double &compression_ratio, size_t &total_compressed_size, Dims ... args) {
    std::vector<size_t> compressed_size;

    total_compressed_size = 0;
    SZ3::uchar *compressed;

    size_t num = 0;
    SZ3::Timer timer_io(true);
    auto data = SZ3::readfile<T>(path, num);
    timer_io.stop("loading from disk");

    {
        std::cout << "****************** compression ****************" << std::endl;
        std::cout << "Interp op          = " << interp_op << std::endl
                  << "Direction          = " << direction_op << std::endl
                  << "Layers             = " << layers << std::endl
                  << "Block size         = " << 0 << std::endl;

        auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};

        auto sz = SZ3::SZProgressiveMQuant<T, N, SZ3::LinearQuantizer2<T>, SZ3::HuffmanEncoder<int>, SZ3::Lossless_zstd>(
                // SZ3::LinearQuantizer2<T>(num, eb, 524288),
                SZ3::LinearQuantizer2<T>(num, 1), // the second arg is dummy.
                SZ3::HuffmanEncoder<int>(),
                // SZ3::ArithmeticEncoder<int>(),
                SZ3::Lossless_zstd(3),
                dims, interp_op, direction_op, 50000, layers, 0
        );
        SZ3::uchar *lossless_data = new SZ3::uchar[size_t((sz.num_elements < 1000000 ? 100 : 2.0) * sz.num_elements) * sizeof(T)]; //?
        sz.setupLayers(data.get());
        SZ3::Timer timer_compress(true);
        timer_compress.start();
        compressed = sz.compress(data.get(), total_compressed_size, lossless_data);
        timer_compress.stop("Compression");

        
        // total_compressed_size = std::accumulate(compressed_size.begin(), compressed_size.end(), (size_t) 0);
        compression_ratio = num * sizeof(T) * 1.0 / total_compressed_size;
        std::cout << "Compressed size = " << total_compressed_size << std::endl;
        std::cout << "Compression ratio = " << compression_ratio << std::endl << std::endl;
    }
    return compressed;
}

template<uint N, typename T, class ... Dims>
T *interp_decompress(const char *path, std::vector<double> & target_ebs, int interp_op, int direction_op,
                                int layers, int mode, SZ3::uchar * compressed, bool writeintoFile, Dims ... args){
    size_t num = 0;
    auto data = SZ3::readfile<T>(path, num);
    T * dec_data = nullptr;

    {
    std::cout << "****************** Decompression ****************" << std::endl;

    auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};
    auto sz = SZ3::SZProgressiveMQuant<T, N, SZ3::LinearQuantizer2<T>, SZ3::HuffmanEncoder<int>, SZ3::Lossless_zstd>(
            // SZ3::LinearQuantizer2<T>(num, eb, 524288),
            SZ3::LinearQuantizer2<T>(num, 1), // the second arg is dummy.
            SZ3::HuffmanEncoder<int>(),
            // SZ3::ArithmeticEncoder<int>(),
            SZ3::Lossless_zstd(),
            dims, interp_op, direction_op, 50000, layers, 0
    );
    sz.setupLayers(data.get());

    // SZ3::Timer timer(true);
    if(mode == 0) {
        dec_data = sz.decompress(compressed, data.get(), target_ebs);
    } else {
        dec_data = sz.decompress_bitrate(compressed, data.get(), target_ebs[0]);
    }

    // timer.stop("Decompression");

    if (writeintoFile){
//        std::string file = std::string(path).substr(std::string(path).rfind('/') + 1) + ".sz3.out";
//        std::cout << "decompressed file = " << file << std::endl;
//        SZ3::writefile(file.c_str(), dec_data.get(), num);
    }

    // if (level_independent <= 0) {
    //     size_t num1 = 0;
    //     auto ori_data = SZ3::readfile<float>(path, num1);
    //     assert(num1 == num);
    //     double psnr, nrmse;
    //     SZ3::verify<float>(ori_data.get(), dec_data, num, psnr, nrmse);
    //     delete[]dec_data;
    //     delete[]compressed;
    // }
//        std::vector<float> error(num);
//        for (size_t i = 0; i < num; i++) {
//            error[i] = ori_data[i] - dec_data[i];
//        }
//        std::string error_file(path);
//        error_file += ".error";
//        SZ3::writefile(error_file.c_str(), error.data(), num);
//        auto compression_ratio = num * sizeof(float) * 1.0 / total_compressed_size;
//        printf("PSNR = %f, NRMSE = %.10G, Compression Ratio = %.2f\n", psnr, nrmse, compression_ratio);
}
    return dec_data;
}
template<uint N, class ... Dims>
double interp_compress_decompress(const char *path, std::vector<double> &target_ebs, int interp_op, int direction_op,
                                int layers, int mode, const char *dataType, Dims ... args) {
    printf("dataType:%s\n", dataType);
    double compression_ratio = -1;
    size_t compressed_size = 0;
    if(dataType[0] == 'f') {
        SZ3::uchar * compressed = interp_compress<N, float>(path, interp_op, direction_op, layers, 
                                                compression_ratio, compressed_size, std::forward<Dims>(args)...);
        float * dec_data = interp_decompress<N, float>(path, target_ebs, interp_op, direction_op, layers, mode,
                                                compressed, false, std::forward<Dims>(args)...);
    } else if(dataType[0] == 'd') {
        SZ3::uchar * compressed = interp_compress<N, double>(path, interp_op, direction_op, layers, 
                                                compression_ratio, compressed_size, std::forward<Dims>(args)...);
        double * dec_data = interp_decompress<N, double>(path, target_ebs, interp_op, direction_op, layers, mode,
                                                compressed, false, std::forward<Dims>(args)...);
    }
    // } else if(dataType[0] == 'I') {
    //     SZ3::uchar * compressed = interp_compress<N, int32_t>(path, interp_op, direction_op, layers, 
    //                                             compression_ratio, std::forward<Dims>(args)...);
    //     int32_t * dec_data = interp_decompress<N, int32_t>(path, target_ebs, interp_op, direction_op, layers, 
    //                                             compressed, false, std::forward<Dims>(args)...);
    // }
    return compression_ratio;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "psz usage: " << argv[0] <<
                  " data_file -[dataType: f/d/I] -num_dim dim0 .. dimn target_eb_num target_abs_eb1 target_abs_eb2 ... [interp_op layers direction_op]"
                  << std::endl
                  << "example: " << argv[0] <<
                  " qmcpack.dat -3 33120 69 69 3 1e-2 1e-3 1e-4 [1 0 3 128]" << std::endl;
        return 0;
    }

    int dim = atoi(argv[3] + 1);
    assert(1 <= dim && dim <= 4);
    int argp = 4;
    std::vector<size_t> dims(dim);
    for (int i = 0; i < dim; i++) {
        dims[i] = atoi(argv[argp++]);
    }
    int target_eb_num = atoi(argv[argp++]);
    
    std::vector<double> target_ebs(target_eb_num);
    for (int i = 0; i < target_eb_num; i++) {
        target_ebs[i] = atof(argv[argp++]);
    }

    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    if (argp < argc) {
        interp_op = atoi(argv[argp++]);
    }
    int layers = 3;
    int mode = 0; // 0: error bound mode; 1: bit rate mode

    if (argp < argc) {
        layers = atoi(argv[argp++]);
    }
    if (argp < argc) {
        direction_op = atoi(argv[argp++]);
    }
    if (argp < argc) {
        mode = atoi(argv[argp++]);
    }
    if (interp_op == -1 || direction_op == -1) {
        std::cout << "Tuning not support.\n";
        return 0;
    }
    std::cout << "[Log] interp_op = " << interp_op << std::endl;
    std::cout << "[Log] direction_op = " << direction_op << std::endl;



    // int block_size = 128;
    // if (argp < argc) {
    //     block_size = atoi(argv[argp++]);
    // }

    std::cout << "[Log] layers = " << layers << std::endl;
    // std::cout << "[Log] block_size = " << block_size << std::endl;
    if (dim == 1) {
        interp_compress_decompress<1>(argv[1], target_ebs, interp_op, direction_op, layers, mode,
                                      argv[2] + 1, dims[0]);
    } else if (dim == 2) {
        interp_compress_decompress<2>(argv[1], target_ebs, interp_op, direction_op, layers, mode,
                                      argv[2] + 1, dims[0], dims[1]);
    } else if (dim == 3) {
        interp_compress_decompress<3>(argv[1], target_ebs, interp_op, direction_op, layers, mode,
                                      argv[2] + 1, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        interp_compress_decompress<4>(argv[1], target_ebs, interp_op, direction_op, layers, mode,
                                      argv[2] + 1, dims[0], dims[1], dims[2], dims[3]);
    }


    return 0;
}
