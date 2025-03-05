#ifndef _DEF_HPP
#define _DEF_HPP

#include <cmath>
#include <vector>

#if __has_include(<boost/align/aligned_allocator.hpp>)
    #include <boost/align/aligned_allocator.hpp>
    template <typename T>
    using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 256>>;
#else
    #include <memory>
    template <typename T>
    using aligned_vector = std::vector<T, std::allocator<T>>;  
#endif

namespace SZ3 {
    typedef unsigned int uint;
    typedef unsigned char uchar;
}

#endif  // _DEF_HPP
