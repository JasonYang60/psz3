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

template<uint N, class ... Dims>
SZ3::uchar *interp_compress(const char *path, int interp_op, int direction_op,
                                int layers, int block_size, double &compression_ratio, Dims ... args) {
    std::vector<size_t> compressed_size;
    size_t total_compressed_size = 0;
    SZ3::uchar *compressed;

    size_t num = 0;
    auto data = SZ3::readfile<float>(path, num);
    {
        std::cout << "****************** compression ****************" << std::endl;
        std::cout << "Interp op          = " << interp_op << std::endl
                  << "Direction          = " << direction_op << std::endl
                  << "Layers             = " << layers << std::endl
                  << "Block size         = " << block_size << std::endl;

        SZ3::Timer timer(true);
        auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};
        auto sz = SZ3::SZProgressiveMQuant<float, N, SZ3::LinearQuantizer2<float>, SZ3::HuffmanEncoder<int>, SZ3::Lossless_zstd>(
                // SZ3::LinearQuantizer2<float>(num, eb, 524288),
                SZ3::LinearQuantizer2<float>(num, 1), // the second arg is dummy.
                SZ3::HuffmanEncoder<int>(),
                // SZ3::ArithmeticEncoder<int>(),
                SZ3::Lossless_zstd(3),
                dims, interp_op, direction_op, 50000, layers, block_size
        );
        compressed = sz.compress(data.get(), total_compressed_size);
        timer.stop("Compression");

        // total_compressed_size = std::accumulate(compressed_size.begin(), compressed_size.end(), (size_t) 0);
        compression_ratio = num * sizeof(float) * 1.0 / total_compressed_size;
        std::cout << "Compressed size = " << total_compressed_size << std::endl;
        std::cout << "Compression ratio = " << compression_ratio << std::endl << std::endl;
    }
    return compressed;
}

template<uint N, class ... Dims>
float *interp_decompress(const char *path, float target_eb, int interp_op, int direction_op,
                                int layers, int block_size, SZ3::uchar * compressed, bool writeintoFile, Dims ... args){
    size_t num = 0;
    auto data = SZ3::readfile<float>(path, num);
    float * dec_data = nullptr;

    {
    std::cout << "****************** Decompression ****************" << std::endl;

    SZ3::Timer timer(true);
    auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};
    auto sz = SZ3::SZProgressiveMQuant<float, N, SZ3::LinearQuantizer2<float>, SZ3::HuffmanEncoder<int>, SZ3::Lossless_zstd>(
            // SZ3::LinearQuantizer2<float>(num, eb, 524288),
            SZ3::LinearQuantizer2<float>(num, 1), // the second arg is dummy.
            SZ3::HuffmanEncoder<int>(),
            // SZ3::ArithmeticEncoder<int>(),
            SZ3::Lossless_zstd(),
            dims, interp_op, direction_op, 50000, layers, block_size
    );
    // dec_data = sz.decompress(compressed, data.get(), target_eb);
    std::vector<float> targetEBs = {1e-2, 1e-3, 1e-4};
    dec_data = sz.decompress(compressed, data.get(), targetEBs);

    timer.stop("Decompression");

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
double interp_compress_decompress(const char *path, float target_eb, int interp_op, int direction_op,
                                int layers, int block_size, Dims ... args) {
    double compression_ratio = -1;
    SZ3::uchar * compressed = interp_compress<N>(path, interp_op, direction_op, layers, block_size, 
                                            compression_ratio, std::forward<Dims>(args)...);
    float * dec_data = interp_decompress<N>(path, target_eb, interp_op, direction_op, layers, block_size, 
                                            compressed, false, std::forward<Dims>(args)...);
    return compression_ratio;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "psz usage: " << argv[0] <<
                  " data_file -num_dim dim0 .. dimn target_abs_eb [interp_op direction_op layers block_size]"
                  << std::endl
                  << "example: " << argv[0] <<
                  " qmcpack.dat -3 33120 69 69 1e-3 [1 0 3 128]" << std::endl;
        return 0;
    }

    int dim = atoi(argv[2] + 1);
    assert(1 <= dim && dim <= 4);
    int argp = 3;
    std::vector<size_t> dims(dim);
    for (int i = 0; i < dim; i++) {
        dims[i] = atoi(argv[argp++]);
    }
    float target_eb = atof(argv[argp++]);

    int interp_op = 0; // linear
    int direction_op = 0; // dimension high -> low
    if (argp < argc) {
        interp_op = atoi(argv[argp++]);
    }
    if (argp < argc) {
        direction_op = atoi(argv[argp++]);
    }
    if (interp_op == -1 || direction_op == -1) {
        std::cout << "Tuning not support.\n";
        return 0;
    }
    std::cout << "[Log] interp_op = " << interp_op << std::endl;
    std::cout << "[Log] direction_op = " << direction_op << std::endl;


    int layers = 3;
    if (argp < argc) {
        layers = atoi(argv[argp++]);
    }

    int block_size = 128;
    if (argp < argc) {
        block_size = atoi(argv[argp++]);
    }

    std::cout << "[Log] layers = " << layers << std::endl;
    std::cout << "[Log] block_size = " << block_size << std::endl;
    if (dim == 1) {
        interp_compress_decompress<1>(argv[1], target_eb, interp_op, direction_op, layers,
                                      block_size, dims[0]);
    } else if (dim == 2) {
        interp_compress_decompress<2>(argv[1], target_eb, interp_op, direction_op, layers,
                                      block_size, dims[0], dims[1]);
    } else if (dim == 3) {
        interp_compress_decompress<3>(argv[1], target_eb, interp_op, direction_op, layers,
                                      block_size, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        interp_compress_decompress<4>(argv[1], target_eb, interp_op, direction_op, layers,
                                      block_size, dims[0], dims[1], dims[2], dims[3]);
    }


    return 0;
}
