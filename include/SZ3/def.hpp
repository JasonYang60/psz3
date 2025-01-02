#ifndef _DEF_HPP
#define _DEF_HPP

#include <cmath>
#include <boost/align/aligned_allocator.hpp>

namespace SZ3 {

    typedef unsigned int uint;
    typedef unsigned char uchar;
    template <typename T>
    using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 256>>;

}


#endif
