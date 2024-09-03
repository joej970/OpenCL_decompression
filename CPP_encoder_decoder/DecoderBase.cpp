#include "DecoderBase.hpp"
#include <stdexcept>

#if(defined(__MINGW32__) || defined(__MINGW64__)) && __cplusplus < 201703L
#    include <experimental/filesystem>
#else
#    include <filesystem>
#endif

#include <cstdint>
#include <vector>


/**********************************
 * Reader class
 *********************************/

Reader::Reader(const std::uint8_t* bitStream, const std::size_t bitStreamSize)
   : m_bitStream(bitStream)
   , m_bitStreamSize(bitStreamSize){};

void Reader::loadFirstByte(const std::uint8_t* bitStream, std::size_t& byteIdx, std::uint8_t& byte)
{
    byte = bitStream[byteIdx];
}

void Reader::loadFirstFourBytes(
   const std::uint8_t* bitStream,
   std::size_t& byteIdx,
   std::uint32_t (*bits_single_bfr)[4],
   std::uint32_t (*bits_shifted_bfr)[4])
{
#ifdef USE_SIMD
    __m128i data = _mm_set_epi8(
       bitStream[byteIdx],
       bitStream[byteIdx + 1],
       bitStream[byteIdx + 2],
       bitStream[byteIdx + 3],
       bitStream[byteIdx + 4],
       bitStream[byteIdx + 5],
       bitStream[byteIdx + 6],
       bitStream[byteIdx + 7],
       bitStream[byteIdx + 8],
       bitStream[byteIdx + 9],
       bitStream[byteIdx + 10],
       bitStream[byteIdx + 11],
       bitStream[byteIdx + 12],
       bitStream[byteIdx + 13],
       bitStream[byteIdx + 14],
       bitStream[byteIdx + 15]);

    for(int i = 31; i >= 0; i--) {
        const static __m128i mask = _mm_set1_epi32(1);
        __m128i right_shift_logical;
        __m128i right_shift_logical_single;
        // *(__uint128_t*)(&bits_shifted_bfr[i]) = (__uint128_t)_mm_srli_epi32(data, i);
        // *(__m128i*)(&bits_shifted_bfr[i]) = _mm_srli_epi32(data, i);
        right_shift_logical = _mm_srli_epi32(data, i);
        // __uint128_t shifted_test;
        // __uint128_t data_test;
        // _mm_store_si128((__m128i*)&shifted_test, right_shift_logical);
        // _mm_store_si128((__m128i*)&data_test, data);
        _mm_store_si128((__m128i*)&bits_shifted_bfr[31 - i], right_shift_logical);
        // std::cout << "After shift ["<< 31-i <<"]: " << std::hex << bits_shifted_bfr[31-i][0] << "'" << bits_shifted_bfr[31-i][1] << "'"  << bits_shifted_bfr[31-i][2] << "'"  << bits_shifted_bfr[31-i][3] << std::dec << std::endl;
        right_shift_logical_single = _mm_and_si128(right_shift_logical, mask);
        _mm_store_si128((__m128i*)&bits_single_bfr[31 - i], right_shift_logical_single);
    }

    // __uint128_t check_mask;
    // _mm_store_si128((__m128i*)&check_mask, mask);

    // __uint128_t long_int = 0;
    // // memcpy(&long_int, bitStream + byteIdx, sizeof(uint32_t));
    // long_int = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
    //            (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
    //            (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
    //            (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
    //            (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
    //            (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
    //            (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
    //            (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];

    // // __uint128_t long_int_simd;
    // // _mm_store_si128((__m128i*)&long_int_simd, data);

    // // if(long_int != long_int_simd) {
    // //     std::cout << "Error: long_int != long_int_simd" << std::endl;
    // // }

    // std::uint32_t bits_shifted_bfr_gt[32][4];
    // std::uint32_t bits_single_bfr_gt[32][4];

    // for(int i = 0; i < 32; i++) {
    //     bits_shifted_bfr_gt[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
    //     bits_shifted_bfr_gt[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
    //     bits_shifted_bfr_gt[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
    //     bits_shifted_bfr_gt[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

    //     bits_single_bfr_gt[i][0] = bits_shifted_bfr_gt[i][0] & 1;
    //     bits_single_bfr_gt[i][1] = bits_shifted_bfr_gt[i][1] & 1;
    //     bits_single_bfr_gt[i][2] = bits_shifted_bfr_gt[i][2] & 1;
    //     bits_single_bfr_gt[i][3] = bits_shifted_bfr_gt[i][3] & 1;
    // }

    // for(int r = 0; r < 32; r++) {
    //     // for(int c = 0; c < 4; c++){
    //     //     // if(bits_single_bfr_gt[r][c] != bits_single_bfr[r][c]){
    //     //     //     std::cout << "Error: bits_single_bfr_gt != bits_single_bfr" << std::endl;
    //     //     //     std::cout << "bits_single_bfr_gt[" << r << "][" << c << "]: " << bits_single_bfr_gt[r][c] << std::endl;
    //     //     //     std::cout << "   bits_single_bfr[" << r << "][" << c << "]: " << bits_single_bfr[r][c] << std::endl;
    //     //     // }
    //     //     if(bits_shifted_bfr_gt[r][c] != bits_shifted_bfr[r][c]){
    //     //         std::cout << "Error: bits_shifted_bfr_gt != bits_shifted_bfr" << std::endl;
    //     //         std::cout << "bits_shifted_bfr_gt[" << r << "][" << c << "]: " << std::hex << bits_shifted_bfr_gt[r][c] << std::endl;
    //     //         std::cout << "bits_shifted_bfr   [" << r << "][" << c << "]: " << std::hex << bits_shifted_bfr[r][c] << std::endl;
    //     //     }
    //     // }
    //     // printf("Row %d bits_shifted_bfr_gt: %x %x %x %x\n", r, bits_shifted_bfr_gt[r][0], bits_shifted_bfr_gt[r][1], bits_shifted_bfr_gt[r][2], bits_shifted_bfr_gt[r][3]);
    //     // printf("Row %d bits_shifted_bfr   : %x %x %x %x\n", r, bits_shifted_bfr[r][0], bits_shifted_bfr[r][1], bits_shifted_bfr[r][2], bits_shifted_bfr[r][3]);
    //     printf("---------\n");
    //     printf(
    //        "Row %d bits_single_bfr_gt : %x %x %x %x\n",
    //        r,
    //        bits_single_bfr_gt[r][0],
    //        bits_single_bfr_gt[r][1],
    //        bits_single_bfr_gt[r][2],
    //        bits_single_bfr_gt[r][3]);
    //     printf(
    //        "Row %d bits_single_bfr    : %x %x %x %x\n",
    //        r,
    //        bits_single_bfr[r][3 - 0],
    //        bits_single_bfr[r][3 - 1],
    //        bits_single_bfr[r][3 - 2],
    //        bits_single_bfr[r][3 - 3]);
    // }

#else
    __uint128_t long_int_temp = 0;
    // memcpy(&long_int, bitStream + byteIdx, sizeof(uint32_t));
    // __uint128_t long_int_gt = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
    //                           (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
    //                           (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
    //                           (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
    //                           (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
    //                           (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
    //                           (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
    //                           (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];

    reverseByteOrder_16bytes(&(bitStream[byteIdx]), (uint8_t*)&long_int_temp);
    __uint128_t long_int = 0;
    memcpy(&long_int, &long_int_temp, sizeof(__uint128_t));
    // __uint128_t long_int;

    for(int i = 0; i < 32; i++) {
        bits_shifted_bfr[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
        bits_shifted_bfr[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
        bits_shifted_bfr[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
        bits_shifted_bfr[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

        bits_single_bfr[i][0] = bits_shifted_bfr[i][0] & 1;
        bits_single_bfr[i][1] = bits_shifted_bfr[i][1] & 1;
        bits_single_bfr[i][2] = bits_shifted_bfr[i][2] & 1;
        bits_single_bfr[i][3] = bits_shifted_bfr[i][3] & 1;
    }

    // printf("bitStream[0:3]: %x %x %x %x\n", bitStream[byteIdx], bitStream[byteIdx + 1], bitStream[byteIdx + 2], bitStream[byteIdx + 3]);
    // printf("      long_int: %x %x %x %x\n", (uint8_t)(long_int >> 24), (uint8_t)(long_int >> 16), (uint8_t)(long_int >> 8), (uint8_t)(long_int));

#endif
}

/**
 * Read last/rightmost/LSB bit at position indicated by ^, from a char* bitStream, 
 * then set cursor to next bit position indicated by '.
 * XXXXXXA____
 * XXXXXXXB___
 * XXXXXXXXC__
 * ________^'_
 */
std::uint32_t Reader::fetchBit_impl(
   const std::uint8_t* bitStream,
   std::size_t bitStream_size,
   std::size_t& bitsReadFromByte,
   std::size_t& byteIdx,
   std::uint8_t& byte)
{
    if(byteIdx == bitStream_size) {
        std::cout << "All bytes have been read." << std::endl;
        throw(BASE_ERROR_ALL_BYTES_ALREADY_READ);
    }
    if(bitsReadFromByte == 8) {
        bitsReadFromByte = 0;
        (byteIdx)++;
        byte = bitStream[byteIdx];   //update only if it all bits have been read
    }
    // todo: consider changing bit order so that there is only bitshift for 1 bit >> 1
    std::uint32_t bit = (byte >> (7 - bitsReadFromByte)) & (std::uint32_t)1;
    bitsReadFromByte++;
    return bit;
}

std::uint32_t Reader::fetchBit_buffered_impl(
   const std::uint8_t* bitStream,
   std::size_t bitStream_size,
   std::size_t& bitsReadFromByte,
   std::size_t& byteIdx,
   std::uint32_t (*bits_single_bfr)[4],
   std::uint32_t (*bits_shifted_bfr)[4])
{
    if(byteIdx == bitStream_size) {
        std::cout << "All bytes have been read." << std::endl;
        throw(BASE_ERROR_ALL_BYTES_ALREADY_READ);
    }
    if(bitsReadFromByte == 128) {
        bitsReadFromByte = 0;
        // (byteIdx)++;
        byteIdx = byteIdx + 16;

#ifdef USE_SIMD
        __m128i data = _mm_set_epi8(
           bitStream[byteIdx],
           bitStream[byteIdx + 1],
           bitStream[byteIdx + 2],
           bitStream[byteIdx + 3],
           bitStream[byteIdx + 4],
           bitStream[byteIdx + 5],
           bitStream[byteIdx + 6],
           bitStream[byteIdx + 7],
           bitStream[byteIdx + 8],
           bitStream[byteIdx + 9],
           bitStream[byteIdx + 10],
           bitStream[byteIdx + 11],
           bitStream[byteIdx + 12],
           bitStream[byteIdx + 13],
           bitStream[byteIdx + 14],
           bitStream[byteIdx + 15]);

        // __m128i data = _mm_load_si128((__m128i*)&bitStream[byteIdx]); // incorrect anyways (byte order is wrong)

        // data = _mm_set_epi64x(bitStream[byteIdx], bitStream[byteIdx + 8]);

        // alignas(16) __uint128_t long_simd_test = 0;
        // _mm_store_si128((__m128i*)&long_simd_test, data);

        // // printf("bitStream[0:3]: %x %x %x %x\n", bitStream[byteIdx], bitStream[byteIdx + 1], bitStream[byteIdx + 2], bitStream[byteIdx + 3]);
        // printf(
        //    "bitStream[0:15]: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x\n",
        //    bitStream[byteIdx],
        //    bitStream[byteIdx + 1],
        //    bitStream[byteIdx + 2],
        //    bitStream[byteIdx + 3],
        //    bitStream[byteIdx + 4],
        //    bitStream[byteIdx + 5],
        //    bitStream[byteIdx + 6],
        //    bitStream[byteIdx + 7],
        //    bitStream[byteIdx + 8],
        //    bitStream[byteIdx + 9],
        //    bitStream[byteIdx + 10],
        //    bitStream[byteIdx + 11],
        //    bitStream[byteIdx + 12],
        //    bitStream[byteIdx + 13],
        //    bitStream[byteIdx + 14],
        //    bitStream[byteIdx + 15]);
        // // printf("      long_int: %x %x %x %x\n", (uint8_t)(long_int >> 24), (uint8_t)(long_int >> 16), (uint8_t)(long_int >> 8), (uint8_t)(long_int));
        // printf(
        //    " long_simd_test: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x\n",
        //    (uint8_t)(long_simd_test >> 120),
        //    (uint8_t)(long_simd_test >> 112),
        //    (uint8_t)(long_simd_test >> 104),
        //    (uint8_t)(long_simd_test >> 96),
        //    (uint8_t)(long_simd_test >> 88),
        //    (uint8_t)(long_simd_test >> 80),
        //    (uint8_t)(long_simd_test >> 72),
        //    (uint8_t)(long_simd_test >> 64),
        //    (uint8_t)(long_simd_test >> 56),
        //    (uint8_t)(long_simd_test >> 48),
        //    (uint8_t)(long_simd_test >> 40),
        //    (uint8_t)(long_simd_test >> 32),
        //    (uint8_t)(long_simd_test >> 24),
        //    (uint8_t)(long_simd_test >> 16),
        //    (uint8_t)(long_simd_test >> 8),
        //    (uint8_t)(long_simd_test));

        for(int i = 31; i >= 0; i--) {
            const static auto mask = _mm_set1_epi32(1);
            __m128i right_shift_logical;
            __m128i right_shift_logical_single;
            // *(__uint128_t*)(&bits_shifted_bfr[i]) = (__uint128_t)_mm_srli_epi32(data, i);
            // *(__m128i*)(&bits_shifted_bfr[i]) = _mm_srli_epi32(data, i);
            right_shift_logical = _mm_srli_epi32(data, i);
            _mm_store_si128((__m128i*)&bits_shifted_bfr[31 - i], right_shift_logical);
            right_shift_logical_single = _mm_and_si128(right_shift_logical, mask);
            _mm_store_si128((__m128i*)&bits_single_bfr[31 - i], right_shift_logical_single);
        }

        // __uint128_t long_int = 0;
        // // memcpy(&long_int, bitStream + byteIdx, sizeof(uint32_t));
        // long_int = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
        //            (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
        //            (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
        //            (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
        //            (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
        //            (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
        //            (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
        //            (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];
        // std::uint32_t bits_shifted_bfr_gt[32][4];
        // std::uint32_t bits_single_bfr_gt[32][4];

        // for(int i = 0; i < 32; i++) {
        //     bits_shifted_bfr_gt[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
        //     bits_shifted_bfr_gt[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
        //     bits_shifted_bfr_gt[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
        //     bits_shifted_bfr_gt[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

        //     bits_single_bfr_gt[i][0] = bits_shifted_bfr_gt[i][0] & 1;
        //     bits_single_bfr_gt[i][1] = bits_shifted_bfr_gt[i][1] & 1;
        //     bits_single_bfr_gt[i][2] = bits_shifted_bfr_gt[i][2] & 1;
        //     bits_single_bfr_gt[i][3] = bits_shifted_bfr_gt[i][3] & 1;
        // }

        // for(int r = 0; r < 32; r++) {
        //     printf("---------\n");
        //     printf(
        //        "Row %d bits_single_bfr_gt : %x %x %x %x\n",
        //        r,
        //        bits_single_bfr_gt[r][0],
        //        bits_single_bfr_gt[r][1],
        //        bits_single_bfr_gt[r][2],
        //        bits_single_bfr_gt[r][3]);
        //     printf(
        //        "Row %d bits_single_bfr    : %x %x %x %x\n",
        //        r,
        //        bits_single_bfr[r][3 - 0],
        //        bits_single_bfr[r][3 - 1],
        //        bits_single_bfr[r][3 - 2],
        //        bits_single_bfr[r][3 - 3]);
        // }

#else
        __uint128_t long_int = 0;
        // memcpy(&long_int, bitStream + byteIdx, sizeof(uint32_t));
        // long_int = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
        //            (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
        //            (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
        //            (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
        //            (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
        //            (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
        //            (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
        //            (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];

        __uint128_t long_int_temp = 0;
        reverseByteOrder_16bytes(
           &(bitStream[byteIdx]),
           (uint8_t*)&long_int_temp);   // for test only to measure timing that could be saved if byte order is reversed
        memcpy(&long_int, &long_int_temp, sizeof(__uint128_t));

        for(int i = 0; i < 32; i++) {
            bits_shifted_bfr[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
            bits_shifted_bfr[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
            bits_shifted_bfr[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
            bits_shifted_bfr[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

            bits_single_bfr[i][0] = bits_shifted_bfr[i][0] & 1;
            bits_single_bfr[i][1] = bits_shifted_bfr[i][1] & 1;
            bits_single_bfr[i][2] = bits_shifted_bfr[i][2] & 1;
            bits_single_bfr[i][3] = bits_shifted_bfr[i][3] & 1;
        }

#endif
    }
// todo: consider changing bit order so that there is only bitshift for 1 bit >> 1
// std::uint32_t bit = (byte >> (7 - bitsReadFromByte)) & (std::uint32_t)1;
#ifdef USE_SIMD
    std::uint32_t bit = bits_single_bfr[bitsReadFromByte % 32][3 - (bitsReadFromByte / 32)];
#else
    std::uint32_t bit = bits_single_bfr[bitsReadFromByte % 32][bitsReadFromByte / 32];
#endif
    bitsReadFromByte++;
    return bit;
}

std::uint32_t Reader::fetchBits_buffered_impl(
   const std::uint8_t* bitStream,
   std::size_t bitStream_size,
   std::size_t& bitsReadFromByte,
   std::size_t& byteIdx,
   std::uint32_t (*bits_single_bfr)[4],
   std::uint32_t (*bits_shifted_bfr)[4],
   std::size_t nBits)
{
    const static std::uint32_t masks[32 + 1] = {
       0,         1,         3,         7,          15,         31,        63,       127,      255,
       511,       1023,      2047,      4095,       8191,       16383,     32767,    65535,    131071,
       262143,    524287,    1048575,   2097151,    4194303,    8388607,   16777215, 33554431, 67108863,
       134217727, 268435455, 536870911, 1073741823, 2147483647, 4294967295};
    if(byteIdx == bitStream_size) {
        std::cout << "All bytes have been read." << std::endl;
        throw(BASE_ERROR_ALL_BYTES_ALREADY_READ);
    }
    if(bitsReadFromByte == 128) {
        bitsReadFromByte = 0;
        // (byteIdx)++;
        byteIdx = byteIdx + 16;

#ifdef USE_SIMD
        __m128i data = _mm_set_epi8(
           bitStream[byteIdx],
           bitStream[byteIdx + 1],
           bitStream[byteIdx + 2],
           bitStream[byteIdx + 3],
           bitStream[byteIdx + 4],
           bitStream[byteIdx + 5],
           bitStream[byteIdx + 6],
           bitStream[byteIdx + 7],
           bitStream[byteIdx + 8],
           bitStream[byteIdx + 9],
           bitStream[byteIdx + 10],
           bitStream[byteIdx + 11],
           bitStream[byteIdx + 12],
           bitStream[byteIdx + 13],
           bitStream[byteIdx + 14],
           bitStream[byteIdx + 15]);

        // __m128i data = _mm_load_si128((__m128i*)&bitStream[byteIdx]);

        for(int i = 31; i >= 0; i--) {
            const static auto mask = _mm_set1_epi32(1);
            __m128i right_shift_logical;
            __m128i right_shift_logical_single;
            // *(__uint128_t*)(&bits_shifted_bfr[i]) = (__uint128_t)_mm_srli_epi32(data, i);
            // *(__m128i*)(&bits_shifted_bfr[i]) = _mm_srli_epi32(data, i);
            right_shift_logical = _mm_srli_epi32(data, i);
            _mm_store_si128((__m128i*)&bits_shifted_bfr[31 - i], right_shift_logical);
            right_shift_logical_single = _mm_and_si128(right_shift_logical, mask);
            _mm_store_si128((__m128i*)&bits_single_bfr[31 - i], right_shift_logical_single);
        }

        // __uint128_t long_int = 0;
        // // memcpy(&long_int, bitStream + byteIdx, sizeof(uint32_t));
        // long_int = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
        //            (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
        //            (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
        //            (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
        //            (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
        //            (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
        //            (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
        //            (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];
        // std::uint32_t bits_shifted_bfr_gt[32][4];
        // std::uint32_t bits_single_bfr_gt[32][4];

        // for(int i = 0; i < 32; i++) {
        //     bits_shifted_bfr_gt[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
        //     bits_shifted_bfr_gt[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
        //     bits_shifted_bfr_gt[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
        //     bits_shifted_bfr_gt[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

        //     bits_single_bfr_gt[i][0] = bits_shifted_bfr_gt[i][0] & 1;
        //     bits_single_bfr_gt[i][1] = bits_shifted_bfr_gt[i][1] & 1;
        //     bits_single_bfr_gt[i][2] = bits_shifted_bfr_gt[i][2] & 1;
        //     bits_single_bfr_gt[i][3] = bits_shifted_bfr_gt[i][3] & 1;
        // }

        // for(int r = 0; r < 32; r++) {
        //     printf("---------\n");
        //     printf(
        //        "Row %d bits_single_bfr_gt : %x %x %x %x\n",
        //        r,
        //        bits_single_bfr_gt[r][0],
        //        bits_single_bfr_gt[r][1],
        //        bits_single_bfr_gt[r][2],
        //        bits_single_bfr_gt[r][3]);
        //     printf(
        //        "Row %d bits_single_bfr    : %x %x %x %x\n",
        //        r,
        //        bits_single_bfr[r][3 - 0],
        //        bits_single_bfr[r][3 - 1],
        //        bits_single_bfr[r][3 - 2],
        //        bits_single_bfr[r][3 - 3]);
        // }

#else
        // uint8_t byte      = bitStream[byteIdx];   //update only if it all bits have been read
        __uint128_t long_int = 0;
        // long_int = (__uint128_t)bitStream[byteIdx] << 120 | (__uint128_t)bitStream[byteIdx + 1] << 112 |
        //            (__uint128_t)bitStream[byteIdx + 2] << 104 | (__uint128_t)bitStream[byteIdx + 3] << 96 |
        //            (__uint128_t)bitStream[byteIdx + 4] << 88 | (__uint128_t)bitStream[byteIdx + 5] << 80 |
        //            (__uint128_t)bitStream[byteIdx + 6] << 72 | (__uint128_t)bitStream[byteIdx + 7] << 64 |
        //            (__uint128_t)bitStream[byteIdx + 8] << 56 | (__uint128_t)bitStream[byteIdx + 9] << 48 |
        //            (__uint128_t)bitStream[byteIdx + 10] << 40 | (__uint128_t)bitStream[byteIdx + 11] << 32 |
        //            (__uint128_t)bitStream[byteIdx + 12] << 24 | (__uint128_t)bitStream[byteIdx + 13] << 16 |
        //            (__uint128_t)bitStream[byteIdx + 14] << 8 | (__uint128_t)bitStream[byteIdx + 15];

        __uint128_t long_int_temp = 0;
        reverseByteOrder_16bytes(
           &(bitStream[byteIdx]),
           (uint8_t*)&long_int_temp);   // for test only to measure timing that could be saved if byte order is reversed
        memcpy(&long_int, &long_int_temp, sizeof(__uint128_t));

        for(int i = 0; i < 32; i++) {
            bits_shifted_bfr[i][0] = (uint32_t)(long_int >> (127 - i - 0 * 32));
            bits_shifted_bfr[i][1] = (uint32_t)(long_int >> (127 - i - 1 * 32));
            bits_shifted_bfr[i][2] = (uint32_t)(long_int >> (127 - i - 2 * 32));
            bits_shifted_bfr[i][3] = (uint32_t)(long_int >> (127 - i - 3 * 32));

            bits_single_bfr[i][0] = bits_shifted_bfr[i][0] & 1;
            bits_single_bfr[i][1] = bits_shifted_bfr[i][1] & 1;
            bits_single_bfr[i][2] = bits_shifted_bfr[i][2] & 1;
            bits_single_bfr[i][3] = bits_shifted_bfr[i][3] & 1;
        }
#endif
    }
    // todo: consider changing bit order so that there is only bitshift for 1 bit >> 1
    // std::uint32_t bit = (byte >> (7 - bitsReadFromByte)) & (std::uint32_t)1;
    std::uint32_t integer;

    // uint32_t availableBits = 32 - bitsReadFromByte;

    // if(bitsReadFromByte > 120) {
    //     std::cout << "Warning: High bit count: " << bitsReadFromByte << std::endl;
    // }

    uint32_t availableBits =
       32 -
       (bitsReadFromByte %
        32);   // 32 because we store 32 bits at a time, 128 because we have 4 32-bit integers and need to read 128 bits before reloading
    if(nBits > availableBits) {
        uint32_t missingBits = nBits - (availableBits);
#ifdef USE_SIMD
        integer = (bits_shifted_bfr[31][3 - (bitsReadFromByte / 32)] & masks[availableBits]) << missingBits;
#else
        integer = (bits_shifted_bfr[31][bitsReadFromByte / 32] & masks[availableBits]) << missingBits;
#endif

        bitsReadFromByte += availableBits;

        integer = integer | Reader::fetchBits_buffered_impl(
                               bitStream,
                               bitStream_size,
                               bitsReadFromByte,
                               byteIdx,
                               bits_single_bfr,
                               bits_shifted_bfr,
                               missingBits);

    } else {
        std::size_t bfr_idx = (nBits - 1) + bitsReadFromByte;
// integer             = bits_shifted_bfr[bfr_idx][bitsReadFromByte / 32] & masks[nBits];
#ifdef USE_SIMD
        integer = bits_shifted_bfr[bfr_idx % 32][3 - ((bfr_idx / 32) % 4)] & masks[nBits];
#else
        integer = bits_shifted_bfr[bfr_idx % 32][((bfr_idx / 32) % 4)] & masks[nBits];
#endif
        bitsReadFromByte += nBits;
    }

    if(bitsReadFromByte > 128) {
        std::cout << "Error: bitsReadFromByte > 128" << std::endl;
    }

    return integer;
}

STATUS_t Reader::getTimestampAndCompressionInfoFromHeader(
   const std::uint8_t* bitStream,
   std::uint64_t& timestamp,
   std::uint64_t& roi,
   std::uint16_t& width,
   std::uint16_t& height,
   std::uint8_t& unaryMaxWidth,
   std::uint8_t& bpp,
   std::uint8_t& lossyBits,
   std::uint8_t& reserved)
{
    // std::cout << "Align of bitStream: " << alignof(bitStream)<<"-byte." << std::endl;
    if(alignof(decltype(bitStream)) != 8) {
        std::cout << "Header can be read only from 8-byte aligned memory." << std::endl;
        return BASE_ERROR_HEADER_NOT_ALIGNED;
    }

    //timestamp = Reader::fetch8bytes(bitStream, 0);
    timestamp = Reader::getTimestamp(bitStream);
    //printf("Decompression Image timestamp: %llu\n", timestamp);

#ifndef ROI_NOT_INCLUDED
    roi = Reader::getRoiHeader(bitStream);
#else
    roi = 0;
#endif
    std::uint64_t header = Reader::getOmlsHeader(bitStream);

    //printf("Decompression Image header: 0x%016llX\n", header);
    // 2-byte aligned
    width         = (reinterpret_cast<const std::uint16_t*>(&header))[0];
    height        = (reinterpret_cast<const std::uint16_t*>(&header))[1];
    unaryMaxWidth = (reinterpret_cast<const std::uint8_t*>(&header))[4];
    bpp           = (reinterpret_cast<const std::uint8_t*>(&header))[5];
    // 1-byte aligned
    lossyBits = (reinterpret_cast<const std::uint8_t*>(&header))[6];
    reserved  = (reinterpret_cast<const std::uint8_t*>(&header))[7];

    //printf("Width: %0d Height: %0d Lossy bits: %0d Unary width: %0d\n", width, height, lossyBits, unaryMaxWidth);

    // Backward compatibility: if bpp is not set, assume 8
    if(bpp == 0) {
        bpp = 8;
    }

    if(width % 16 != 0 || height % 16 != 0 || (bpp != 8 && bpp != 10 && bpp != 12)) {
        return BASE_ERROR_HEADER_DATA_INVALID;
    }

    return BASE_SUCCESS;
}

std::uint64_t Reader::fetch8bytes(const std::uint8_t* bitStream, std::size_t byteOffset)
{
    // std::cout << "Align of bitStream: " << alignof(decltype(bitStream)) << "-byte." << std::endl;
    if(alignof(decltype(bitStream)) != 8) {
        std::cout << "Header can be read only from 8-byte aligned memory." << std::endl;
        std::cerr << "Header can be read only from 8-byte aligned memory." << std::endl;
    }
    auto p_header = reinterpret_cast<const std::uint64_t*>(&bitStream[byteOffset]);

    return *p_header;
};

/**********************************
 * Misc functions
 *********************************/

/**
     * Inverts byte order. Used to calculate how much time could be saved if byte order is changed on FPGA.
     */
// inline void reverseByteOrder_16bytes(const std::uint8_t* bitStream, std::uint8_t* dest)
void reverseByteOrder_16bytes(const std::uint8_t* bitStream, std::uint8_t* dest)
{
    const int bytes = 16;
    for(std::size_t i = 0; i < bytes; i++) {
        dest[i] = bitStream[bytes - i - 1];
    }
}

/**********************************
 * DecoderBase class
 *********************************/

std::int16_t DecoderBase::fromAbs(std::uint16_t absValue)
{
    if(absValue % 2 == 0) {
        return absValue / 2;
    } else {
        return -((absValue + 1) / 2);
    }
};

/**
 * Reads bitstream from fileName. Interprets first 4 bytes as image resolution.
 * The pointer to the bitstream data is then assigned to member m_pFileData.
*/
STATUS_t DecoderBase::importBitstream(const char* fileName, std::unique_ptr<std::vector<std::uint8_t>>& outData)
{
    std::ifstream rf(fileName, std::ios::in | std::ios::binary);
    if(!rf) {
        return BASE_CANNOT_OPEN_INPUT_FILE;
    }

    std::vector<std::uint8_t> inputBytes(
       (std::istreambuf_iterator<char>(rf)),   //
       (std::istreambuf_iterator<char>()));

    outData =
       std::move(std::make_unique<std::vector<std::uint8_t>>(std::forward<std::vector<std::uint8_t>>(inputBytes)));

    rf.close();

    return BASE_SUCCESS;
};

STATUS_t DecoderBase::handleReturnValue(STATUS_t status)
{
    switch(status) {
        case BASE_SUCCESS:
            //std::cout << "Operation successful." << std::endl;
            break;
        case BASE_ERROR:
            std::cout << "DecoderBase: General error." << std::endl;
            break;
        case BASE_DIVISION_ERROR:
            std::cout << "DecoderBase: Division error." << std::endl;
            break;
        case BASE_ERROR_ALL_BYTES_ALREADY_READ:
            std::cout << "DecoderBase: All bytes already read error." << std::endl;
            break;
        case BASE_CANNOT_OPEN_INPUT_FILE:
            std::cout << "DecoderBase: Cannot open input file error." << std::endl;
            break;
        case BASE_CANNOT_OPEN_OUTPUT_FILE:
            std::cout << "DecoderBase: Cannot open output file error." << std::endl;
            break;
        case BASE_OUTPUT_BUFFER_FALSE_SIZE:
            std::cout << "DecoderBase: Output buffer false size error." << std::endl;
            break;
        case BASE_ERROR_HEADER_NOT_ALIGNED:
            std::cout << "DecoderBase: Header not aligned error." << std::endl;
            break;
        case BASE_ERROR_HEADER_DATA_INVALID:
            std::cout << "Invalid header values. Received image is corrupted. Might be camera image buffer issue. Try "
                         "increasing buffer size."
                      << std::endl;
            break;
        case BASE_OPENCL_ERROR:
            std::cout << "DecoderBase: OpenCL error." << std::endl;
            break;
        default:
            std::cout << "DecoderBase: Unknown error." << std::endl;
            break;
    }
    return status;
}

void createMissingDirectory(const char* folder_out)
{
    /** Create missing directories*/
#if(defined(__MINGW32__) || defined(__MINGW64__)) && __cplusplus < 201703L
    using namespace std::experimental::filesystem;
#else
    using namespace std::filesystem;
#endif
    char out_path[500];

    if(!path(folder_out).is_absolute()) {
        sprintf_s(out_path, "./%s", folder_out);
    } else {
        sprintf_s(out_path, "%s", folder_out);
    };
    //}

    //std::cout << "Proposed out location is: " << out_path << std::endl;
    if(!exists(out_path)) {
        if(create_directory(out_path)) {
            //std::cout << "Created new directory: " << out_path << std::endl;
        } else {
            std::cout << "Failed to create directory: " << out_path << std::endl;
        }
    }
    {
        //std::cout << "Directory already exists: " << out_path << std::endl;
    }
}
