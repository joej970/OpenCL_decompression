#pragma once

#include "globalDefines.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef INCLUDE_OPENCL
#    define CL_TARGET_OPENCL_VERSION 200
#    include "OpenCL_sources/opencl_platforms.hpp"
#    include <CL/cl.h>
#    include <emmintrin.h>   // For SIMD operations
#    include <immintrin.h>   // For SIMD operations

#endif

#ifdef TIMING_EN
#    include <chrono>
#endif

#define USE_SIMD
// #define USE_LUT

void createMissingDirectory(const char* folder_out);

typedef uint32_t STATUS_t;
// Define a macro to check the return value of a function
#define RETURN_ON_FAILURE(func)      \
    do {                             \
        STATUS_t returnValue = func; \
        if(returnValue > 0) {        \
            return returnValue;      \
        }                            \
    } while(0);

enum
{
    BASE_SUCCESS                      = 0,
    BASE_ERROR                        = 1,
    BASE_DIVISION_ERROR               = 2,
    BASE_ERROR_ALL_BYTES_ALREADY_READ = 3,
    BASE_CANNOT_OPEN_INPUT_FILE       = 4,
    BASE_CANNOT_OPEN_OUTPUT_FILE      = 5,
    BASE_OUTPUT_BUFFER_FALSE_SIZE     = 6,
    BASE_ERROR_HEADER_NOT_ALIGNED     = 10,
    BASE_ERROR_HEADER_DATA_INVALID    = 11,
    BASE_OPENCL_ERROR                 = 12,
};

struct lut_data_t {
    std::uint16_t quotient;
    bool q_done;
    std::uint16_t remainder;
    std::uint8_t r_count;
    std::uint16_t truncated_coding_value;
    std::uint8_t t_count;
    std::uint8_t offset_o;
};

/*
* Struct that manages bitstream reading. It keeps internal state of bitstream pointer.
*/
struct Reader {
    // const std::vector<std::uint8_t>& m_bitStream;
    const std::uint8_t* m_bitStream = nullptr;
    std::size_t m_bitStreamSize     = 0;
    std::size_t m_bitsReadFromByte  = 0;
    std::size_t m_byteIdx           = 0;
    std::uint8_t m_byte             = 0;

    alignas(16) std::uint32_t m_bits_single_bfr[32][4]  = {0};
    alignas(16) std::uint32_t m_bits_shifted_bfr[32][4] = {0};
    /*
    * Reader constructor.
    */
    Reader(const std::uint8_t* bitStream, const std::size_t bitStreamSize);

    /*
    * Fetches single bit.
    */
    static std::uint32_t fetchBit_impl(
       const std::uint8_t* bitStream,
       std::size_t bitStream_size,
       std::size_t& bitsReadFromByte,
       std::size_t& byteIdx,
       std::uint8_t& byte);

    /*
    * Fetches single bit.
    */
    static std::uint32_t fetchBit_buffered_impl(
       const std::uint8_t* bitStream,
       std::size_t bitStream_size,
       std::size_t& bitsReadFromByte,
       std::size_t& byteIdx,
       std::uint32_t (*bits_single_bfr)[4],
       std::uint32_t (*bits_shifted_bfr)[4]);

    /*
    * Fetches nBits from bitstream.
    */
    static std::uint32_t fetchBits_buffered_impl(
       const std::uint8_t* bitStream,
       std::size_t bitStream_size,
       std::size_t& bitsReadFromByte,
       std::size_t& byteIdx,
       std::uint32_t (*bits_single_bfr)[4],
       std::uint32_t (*bits_shifted_bfr)[4],
       std::size_t nBits);

    /*
    * Fetches 8 bytes from bitstream[byteOffset:byteOffset+8].
    */
    static std::uint64_t fetch8bytes(   //
       const std::uint8_t* bitStream,
       std::size_t byteOffset);

    /*
    * Loads first byte from bitstream to m_byte and increments bitstream pointer by 1 byte.
    */
    void loadFirstByte(const std::uint8_t* bitStream, std::size_t& byteIdx, std::uint8_t& byte);

    /**
     * Loads first 4 bytes from bitstream and increments bitstream pointer by 4 bytes.
     */
    void loadFirstFourBytes(
       const std::uint8_t* bitStream,
       std::size_t& byteIdx,
       std::uint32_t (*bits_single_bfr)[4],
       std::uint32_t (*bits_shifted_bfr)[4]);

    /*
    * Loads first byte from bitstream member object.
    */
    void loadFirstByte()
    {
        loadFirstByte(m_bitStream, m_byteIdx, m_byte);
    }

    /**
     * Loads first 4 bytes from bitstream member object.
     */
    void loadFirstFourBytes()
    {
        loadFirstFourBytes(m_bitStream, m_byteIdx, m_bits_single_bfr, m_bits_shifted_bfr);
    }

    std::uint32_t fetchBit()
    {
        return Reader::fetchBit_impl(m_bitStream, m_bitStreamSize, m_bitsReadFromByte, m_byteIdx, m_byte);
    };

    std::uint32_t fetchBit_buffered()
    {
        // return Reader::fetchBit(m_bitStream, m_bitStreamSize, m_bitsReadFromByte, m_byteIdx, m_byte);
        return Reader::fetchBit_buffered_impl(
           m_bitStream,
           m_bitStreamSize,
           m_bitsReadFromByte,
           m_byteIdx,
           m_bits_single_bfr,
           m_bits_shifted_bfr);
    };

    std::uint32_t fetchBits_buffered(std::size_t nBits)
    {
        return Reader::fetchBits_buffered_impl(
           m_bitStream,
           m_bitStreamSize,
           m_bitsReadFromByte,
           m_byteIdx,
           m_bits_single_bfr,
           m_bits_shifted_bfr,
           nBits);
    };

    /*
    * Gets timestamp from bitstream[0:7]
    */
    static std::uint64_t getTimestamp(const std::uint8_t* bitStream)
    {
        return Reader::fetch8bytes(bitStream, 0);
    }

    /*
    * Gets roi header from bitstream[8:15]
    */
    static std::uint64_t getRoiHeader(const std::uint8_t* bitStream)
    {
        return Reader::fetch8bytes(bitStream, 8);
    }

    /*
    * Gets roi header from bitstream[8:15]
    */
    static std::uint64_t getOmlsHeader(const std::uint8_t* bitStream)
    {
#ifdef ROI_NOT_INCLUDED
        return Reader::fetch8bytes(bitStream, 8);
#else
        return Reader::fetch8bytes(bitStream, 16);
#endif
    }

    /*
    * Gets timestamp from bitstream[0:7] and header from bitstream[8:15]
    */
    static STATUS_t getTimestampAndCompressionInfoFromHeader(
       const std::uint8_t* bitStream,
       std::uint64_t& timestamp,
       std::uint64_t& roi,
       std::uint16_t& width,
       std::uint16_t& height,
       std::uint8_t& unaryMaxWidth,
       std::uint8_t& bpp,
       std::uint8_t& lossyBits,
       std::uint8_t& reserved);
};

/**
     * Translates to MSB first and shifts to get actual value. This function will not be needed if remainders are encoded in MSB first.
     */
inline std::uint32_t getMSBfirst(std::uint32_t lsb, std::size_t validBits)
{
    std::uint32_t msb = 0;
    for(std::size_t i = 0; i < validBits; i++) {
        msb |= ((lsb >> i) & 1) << (validBits - 1 - i);
    }
    // msb >>= (16 - validBits);
    return msb;
}

/**
     * Inverts byte order. Used to calculate how much time could be saved if byte order is changed on FPGA.
     */
inline void reverseByteOrder_16bytes(const std::uint8_t* bitStream, std::uint8_t* dest);

struct DecoderBase {

    /**
 * @param width_a and @param height_a are full image width and height.
 * BayerCFA image.
 */
    template<typename T>
    static STATUS_t decodeBitstreamParallel_actual(
       std::size_t width_a,
       std::size_t height_a,
       std::size_t lossyBits_a,
       std::size_t unaryMaxWidth_a,
       std::size_t bpp_a,
       const std::uint8_t* bitStream,
       const std::size_t bitStreamSize,
       T* bayerGB,
       std::size_t bayerGBSize)
    {
        std::uint32_t N_threshold = 8;
        std::uint32_t A_init      = 32;

        Reader reader{bitStream, bitStreamSize};

        std::size_t width;
        std::size_t height;
        std::size_t lossyBits;
        std::size_t unaryMaxWidth;
        unaryMaxWidth        = unaryMaxWidth_a;
        width                = width_a / 2;
        height               = height_a / 2;
        lossyBits            = lossyBits_a;
        unaryMaxWidth        = unaryMaxWidth_a;
        std::uint32_t k_seed = bpp_a + 3;   // max 12 BPP + 3 = 15

        // Check if full decompressed image buffer is large enough. Multply by 4 because there are Gb,B,R & Gr channels.
        if(2 * width * 2 * height != bayerGBSize) {
            fprintf(
               stdout,
               "DecoderBase: expected size of output buffer: %zu, actual size: %zu (bpp: %zu)\n",
               2 * width * 2 * height,
               bayerGBSize,
               bpp_a);
            return BASE_OUTPUT_BUFFER_FALSE_SIZE;
        }

        reader.loadFirstByte();

        std::uint32_t A[]        = {A_init, A_init, A_init, A_init};
        std::uint32_t N          = N_START;
        std::int16_t YCCC[]      = {0, 0, 0, 0};
        std::int16_t YCCC_prev[] = {0, 0, 0, 0};
        std::int16_t YCCC_up[]   = {0, 0, 0, 0};

        // 1.) decode 4 seed pixels
        // Seed pixel
        std::uint16_t posValue[] = {0, 0, 0, 0};
        for(std::size_t ch = 0; ch < 4; ch++) {
            // for(std::uint32_t n = k_SEED + 1; n > 0; n--) { // MSB first
            //     std::uint32_t lastBit = fetchBit(reader);
            //     posValue[ch]          = (posValue[ch] << 1) | lastBit;
            // }
            (void)reader.fetchBit();   // read delimiter
            for(std::uint32_t n = 0; n < k_seed; n++) {   // LSB first
                std::uint32_t lastBit = reader.fetchBit();
                posValue[ch]          = posValue[ch] | ((std::uint16_t)lastBit << n);
            };
            YCCC[ch] = DecoderBase::fromAbs(posValue[ch]);
        }

        for(std::size_t ch = 0; ch < 4; ch++) {   //
            A[ch] += YCCC[ch] > 0 ? YCCC[ch] : -YCCC[ch];
        }

        RETURN_ON_FAILURE(DecoderBase::YCCC_to_BayerGB<T>(
           YCCC[0],
           YCCC[1],
           YCCC[2],
           YCCC[3],
           bayerGB[0],
           bayerGB[1],
           bayerGB[2 * width],
           bayerGB[2 * width + 1],
           lossyBits))

        memcpy(YCCC_up, YCCC, sizeof(YCCC));
        memcpy(YCCC_prev, YCCC, sizeof(YCCC));

        for(std::size_t idx = 1; idx < height * width; idx++) {

            // 2.) AGOR
            std::uint16_t quotient[]  = {0, 0, 0, 0};
            std::uint16_t remainder[] = {0, 0, 0, 0};
            std::uint16_t k[]         = {0, 0, 0, 0};
            std::int16_t dpcm_curr[]  = {0, 0, 0, 0};

            for(std::size_t ch = 0; ch < 4; ch++) {
                // for(std::size_t it = 0; it < 10; it++) {
                for(std::size_t it = 0; it < (bpp_a + 2); it++) {
                    if((N << k[ch]) < A[ch]) {
                        k[ch] = k[ch] + 1;
                    } else {
                        k[ch] = k[ch];
                    }
                }
                std::uint32_t lastBit;

                std::uint16_t absVal = 0;
                do {   // decode quotient: unary coding
                    lastBit = reader.fetchBit();
                    if(lastBit == 1) {   //
                        quotient[ch]++;
                    }
                } while(lastBit == 1 && quotient[ch] < unaryMaxWidth);

                /* m_unaryMaxWidth * '1' -> indicates binary coding of positive value */
                if(quotient[ch] >= unaryMaxWidth) {
                    // for(std::uint32_t n = 0; n < k_MAX; n++) {   //decode remainder
                    for(std::uint32_t n = 0; n < k_seed; n++) {   //decode remainder
                        lastBit = reader.fetchBit();
                        absVal  = absVal | ((std::uint16_t)lastBit << n); /* LSB first */
                        // absVal  = (absVal << 1) | lastBit; /* MSB first*/
                    };
                    // absVal = reader.fetchBits_buffered(k_seed);

                } else {
                    for(std::uint32_t n = 0; n < k[ch]; n++) {   //decode remainder
                        lastBit       = reader.fetchBit();
                        remainder[ch] = remainder[ch] | ((std::uint16_t)lastBit << n); /* LSB first*/
                        // remainder[ch] = (remainder[ch] << 1) | lastBit; /* MSB first*/
                    };
                    // remainder[ch] = reader.fetchBits_buffered(k[ch]);
                    absVal = (quotient[ch] << k[ch]) + remainder[ch];
                }
                dpcm_curr[ch] = DecoderBase::fromAbs(absVal);

                A[ch] += dpcm_curr[ch] > 0 ? dpcm_curr[ch] : -dpcm_curr[ch];
                YCCC[ch] = YCCC_prev[ch] + dpcm_curr[ch];
                if(idx % width == 0) {   // when first column pixel, use pixel one row up as a reference instead
                    YCCC[ch]    = YCCC_up[ch] + dpcm_curr[ch];
                    YCCC_up[ch] = YCCC[ch];
                }
                YCCC_prev[ch] = YCCC[ch];   // save pixel to be used as a reference for the next pixel
            }

            std::size_t idxGB = (idx / width) * width * 4 + 2 * (idx % width);
            RETURN_ON_FAILURE(DecoderBase::YCCC_to_BayerGB<T>(
                                 YCCC[0],   //
                                 YCCC[1],
                                 YCCC[2],
                                 YCCC[3],
                                 bayerGB[idxGB],
                                 bayerGB[idxGB + 1],
                                 bayerGB[idxGB + 2 * width],
                                 bayerGB[idxGB + 2 * width + 1],
                                 lossyBits);)
            N += 1;
            if(N >= N_threshold) {
                N >>= 1;
                A[0] >>= 1;
                A[1] >>= 1;
                A[2] >>= 1;
                A[3] >>= 1;
            }
            A[0] = A[0] < A_MIN ? A_MIN : A[0];
            A[1] = A[1] < A_MIN ? A_MIN : A[1];
            A[2] = A[2] < A_MIN ? A_MIN : A[2];
            A[3] = A[3] < A_MIN ? A_MIN : A[3];
        }
        return BASE_SUCCESS;
    }

    /**
 * @param width_a and @param height_a are related to channel size which is one half of the actual BayerCFA image.
 * BayerCFA image.
 */
    STATUS_t decodeBitstreamParallel_pseudo_gpu(
       std::size_t width_a,
       std::size_t height_a,
       std::size_t lossyBits_a,
       std::size_t unaryMaxWidth_a,
       std::size_t bpp_a,
       const std::uint8_t* bitStream,
       const std::size_t bitStreamSize,
       std::uint8_t* bayerGB,
       std::size_t bayerGBSize)
    {

        printf("Pseudo GPU decoding started!\n");

        if(bpp_a != 8) {
            fprintf(
               stdout,
               "DecoderBase: bpp is not 8, but %zu. GPU decompression implemented only for 8 BPP.\n",
               bpp_a);
            return BASE_ERROR;
        }

        std::uint32_t N_threshold = 8;
        std::uint32_t A_init      = 32;

        Reader reader{bitStream, bitStreamSize};

        std::size_t width;
        std::size_t height;
        std::size_t lossyBits;
        std::size_t unaryMaxWidth;
        unaryMaxWidth        = unaryMaxWidth_a;
        width                = width_a / 2;   // half size of the actual BayerCFA image
        height               = height_a / 2;
        lossyBits            = lossyBits_a;
        unaryMaxWidth        = unaryMaxWidth_a;
        std::uint32_t k_seed = bpp_a + 3;   // max 12 BPP + 3 = 15

        // Check if full decompressed image buffer is large enough. Multply by 4 because there are Gb,B,R & Gr channels.
        if(2 * width * 2 * height != bayerGBSize) {
            fprintf(
               stdout,
               "DecoderBase: expected size of output buffer: %zu, actual size: %zu (bpp: %zu)\n",
               2 * width * 2 * height,
               bayerGBSize,
               bpp_a);
            return BASE_OUTPUT_BUFFER_FALSE_SIZE;
        }

        reader.loadFirstByte();

#ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin         = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point begin_parsing = std::chrono::steady_clock::now();
#endif

        std::uint32_t A[] = {A_init, A_init, A_init, A_init};
        // std::uint32_t N   = N_START;
        std::uint32_t N = N_START - 1;
        //std::int16_t YCCC[]      = {0, 0, 0, 0};
        std::vector<std::int16_t> YCCC(4 * (height * width));
        std::vector<std::int16_t> YCCC_prev_debug(4 * (height * width));
        // std::int16_t YCCC_prev[] = {0, 0, 0, 0};
        //std::int16_t YCCC_up[]   = {0, 0, 0, 0};

        // 1.) decode 4 seed pixels

        std::vector<std::int16_t> YCCC_dpcm(4 * (height * width));

        // 2.) AGOR
        // this will cause switch to unary coding and will decode seed pixel appropriatelly
        std::uint16_t quotient[] = {
           (std::uint16_t)unaryMaxWidth,
           (std::uint16_t)unaryMaxWidth,
           (std::uint16_t)unaryMaxWidth,
           (std::uint16_t)unaryMaxWidth};
        std::uint16_t remainder[] = {0, 0, 0, 0};
        std::uint16_t k[]         = {0, 0, 0, 0};
        std::uint16_t absVal[]    = {0, 0, 0, 0};
        // Get DPCM
        for(std::size_t idx = 0; idx < height * width; idx++) {
            // for(std::size_t idx = 1; idx < height * width; idx++) {

            for(std::size_t ch = 0; ch < 4; ch++) {
                for(std::size_t it = 0; it < (bpp_a + 2); it++) {
                    if((N << k[ch]) < A[ch]) {
                        k[ch] = k[ch] + 1;
                    } else {
                        k[ch] = k[ch];
                    }
                }
                std::uint32_t lastBit;

                do {   // decode quotient: unary coding
                    lastBit = reader.fetchBit();
                    if(lastBit == 1) {   //
                        quotient[ch]++;
                    }
                } while(lastBit == 1 && quotient[ch] < unaryMaxWidth);

                /* m_unaryMaxWidth * '1' -> indicates binary coding of positive value */
                if(quotient[ch] >= unaryMaxWidth) {
                    for(std::uint32_t n = 0; n < k_seed; n++) {   //decode remainder
                        lastBit    = reader.fetchBit();
                        absVal[ch] = absVal[ch] | ((std::uint16_t)lastBit << n); /* LSB first */
                        // absVal  = (absVal << 1) | lastBit; /* MSB first*/
                    };
                } else {
                    for(std::uint32_t n = 0; n < k[ch]; n++) {   //decode remainder
                        lastBit       = reader.fetchBit();
                        remainder[ch] = remainder[ch] | ((std::uint16_t)lastBit << n); /* LSB first*/
                        // remainder[ch] = (remainder[ch] << 1) | lastBit; /* MSB first*/
                    };
                    absVal[ch] = quotient[ch] * (1 << k[ch]) + remainder[ch];
                }

                // TODO: OPPORTUNITY: unsigned absVal to signed dpcm can also be done in parallel
            }
            YCCC_dpcm[4 * idx + 0] = DecoderBase::fromAbs(absVal[0]);
            YCCC_dpcm[4 * idx + 1] = DecoderBase::fromAbs(absVal[1]);
            YCCC_dpcm[4 * idx + 2] = DecoderBase::fromAbs(absVal[2]);
            YCCC_dpcm[4 * idx + 3] = DecoderBase::fromAbs(absVal[3]);

            A[0] += YCCC_dpcm[4 * idx + 0] > 0 ? YCCC_dpcm[4 * idx + 0] : -YCCC_dpcm[4 * idx + 0];
            A[1] += YCCC_dpcm[4 * idx + 1] > 0 ? YCCC_dpcm[4 * idx + 1] : -YCCC_dpcm[4 * idx + 1];
            A[2] += YCCC_dpcm[4 * idx + 2] > 0 ? YCCC_dpcm[4 * idx + 2] : -YCCC_dpcm[4 * idx + 2];
            A[3] += YCCC_dpcm[4 * idx + 3] > 0 ? YCCC_dpcm[4 * idx + 3] : -YCCC_dpcm[4 * idx + 3];

            N += 1;
            if(N >= N_threshold) {
                N >>= 1;
                A[0] >>= 1;
                A[1] >>= 1;
                A[2] >>= 1;
                A[3] >>= 1;
            }

            quotient[0]  = 0;
            quotient[1]  = 0;
            quotient[2]  = 0;
            quotient[3]  = 0;
            remainder[0] = 0;
            remainder[1] = 0;
            remainder[2] = 0;
            remainder[3] = 0;
            k[0]         = 0;
            k[1]         = 0;
            k[2]         = 0;
            k[3]         = 0;
            absVal[0]    = 0;
            absVal[1]    = 0;
            absVal[2]    = 0;
            absVal[3]    = 0;
        }

        YCCC[0] = YCCC_dpcm[0];
        YCCC[1] = YCCC_dpcm[1];
        YCCC[2] = YCCC_dpcm[2];
        YCCC[3] = YCCC_dpcm[3];

#ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_parsing = std::chrono::steady_clock::now();
#endif

        // calculate all pixels in first column
        // CALCULATE ALL YCCC from DPCM
        for(std::size_t row = 1; row < height; row++) {
            auto idx_curr =
               row *
               (4 *
                width);   // 4*width (because there are 4 channels of each width (which is half of the actual image width))
            auto idx_prev = (row - 1) * (4 * width);

            YCCC_prev_debug[idx_curr + 0] = YCCC[idx_prev + 0];
            YCCC_prev_debug[idx_curr + 1] = YCCC[idx_prev + 1];
            YCCC_prev_debug[idx_curr + 2] = YCCC[idx_prev + 2];
            YCCC_prev_debug[idx_curr + 3] = YCCC[idx_prev + 3];
            YCCC[idx_curr + 0]            = YCCC[idx_prev + 0] + YCCC_dpcm[idx_curr + 0];
            YCCC[idx_curr + 1]            = YCCC[idx_prev + 1] + YCCC_dpcm[idx_curr + 1];
            YCCC[idx_curr + 2]            = YCCC[idx_prev + 2] + YCCC_dpcm[idx_curr + 2];
            YCCC[idx_curr + 3]            = YCCC[idx_prev + 3] + YCCC_dpcm[idx_curr + 3];
        }

#ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_dpcm_to_yccc = std::chrono::steady_clock::now();
#endif

        // calculate all pixels in each row
        for(std::size_t row = 0; row < height; row++) {

            auto row_offset = row * (4 * width);
            for(std::size_t col = 1; col < width; col++) {
                auto idx_curr                 = row_offset + 4 * col;
                auto idx_prev                 = row_offset + 4 * (col - 1);
                YCCC_prev_debug[idx_curr + 0] = YCCC[idx_prev + 0];
                YCCC_prev_debug[idx_curr + 1] = YCCC[idx_prev + 1];
                YCCC_prev_debug[idx_curr + 2] = YCCC[idx_prev + 2];
                YCCC_prev_debug[idx_curr + 3] = YCCC[idx_prev + 3];
                YCCC[idx_curr + 0]            = YCCC[idx_prev + 0] + YCCC_dpcm[idx_curr + 0];
                YCCC[idx_curr + 1]            = YCCC[idx_prev + 1] + YCCC_dpcm[idx_curr + 1];
                YCCC[idx_curr + 2]            = YCCC[idx_prev + 2] + YCCC_dpcm[idx_curr + 2];
                YCCC[idx_curr + 3]            = YCCC[idx_prev + 3] + YCCC_dpcm[idx_curr + 3];
            }
        }

#ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_dpcm_to_yccc    = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point begin_yccc_to_bayer = std::chrono::steady_clock::now();
#endif

        // TODO: OPPORTUNITY: can be done in parallel
        // for(std::size_t idx = 1; idx < height * width; idx++) {
        for(std::size_t idx = 0; idx < height * width; idx++) {
            std::size_t idxGB = (idx / width) * width * 4 + 2 * (idx % width);
            RETURN_ON_FAILURE(DecoderBase::YCCC_to_BayerGB(
                                 YCCC[4 * idx + 0],   //
                                 YCCC[4 * idx + 1],
                                 YCCC[4 * idx + 2],
                                 YCCC[4 * idx + 3],
                                 bayerGB[idxGB],
                                 bayerGB[idxGB + 1],
                                 bayerGB[idxGB + 2 * width],
                                 bayerGB[idxGB + 2 * width + 1],
                                 lossyBits);)
        }

        // char outputFile[200];
        // char txt[200];
        // sprintf(outputFile, "C:/DATA/Repos/OMLS_Masters_SW/images/lite_dataset/dump/gpu_DPCM.txt");
        // std::ofstream wf(outputFile, std::ios::out);
        // if(!wf) {
        //     char msg[200];
        //     sprintf(msg, "Cannot open specified file: %s", outputFile);
        //     throw std::runtime_error(msg);
        // }
        // std::sprintf(txt, "%zu %zu pred YCCC, curr YCCC, dpcm\n", width, height);
        // wf << txt;
        // int16_t mask_Y      = (1 << (bpp_a + 2)) - 1;   // 0x03FF
        // int16_t mask_C      = (1 << (bpp_a + 1)) - 1;   // 0x01FF;
        // int16_t mask_Y_dpcm = (1 << (bpp_a + 3)) - 1;   // 0x07FF;
        // int16_t mask_C_dpcm = (1 << (bpp_a + 2)) - 1;   // 0x03FF;

        // for(size_t i = 0; i < width * height; i++) {
        //     std::sprintf(
        //        txt,
        //        "%4zu %4zu : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX\n",
        //        (size_t)i / width,
        //        (size_t)i % width,
        //        (YCCC_prev_debug[4 * i + 0] & mask_Y),
        //        (YCCC_prev_debug[4 * i + 1] & mask_C),
        //        (YCCC_prev_debug[4 * i + 2] & mask_C),
        //        (YCCC_prev_debug[4 * i + 3] & mask_C),
        //        (YCCC[4 * i + 0] & mask_Y),
        //        (YCCC[4 * i + 1] & mask_C),
        //        (YCCC[4 * i + 2] & mask_C),
        //        (YCCC[4 * i + 3] & mask_C),
        //        (YCCC_dpcm[4 * i + 0] & mask_Y_dpcm),
        //        (YCCC_dpcm[4 * i + 1] & mask_C_dpcm),
        //        (YCCC_dpcm[4 * i + 2] & mask_C_dpcm),
        //        (YCCC_dpcm[4 * i + 3] & mask_C_dpcm));
        //     wf << txt;
        // }
        // wf.close();

#ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_yccc_to_bayer = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point end               = std::chrono::steady_clock::now();
        std::cout << "Pseudo GPU: Parallel decoding time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        std::cout << "Pseudo GPU: Parsing time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parsing - begin_parsing).count()
                  << "[ms]" << std::endl;
        std::cout
           << "Pseudo GPU: DPCM to YCCC time = "
           << std::chrono::duration_cast<std::chrono::milliseconds>(end_dpcm_to_yccc - begin_dpcm_to_yccc).count()
           << "[ms]" << std::endl;
        std::cout
           << "Pseudo GPU: YCCC to Bayer time = "
           << std::chrono::duration_cast<std::chrono::milliseconds>(end_yccc_to_bayer - begin_yccc_to_bayer).count()
           << "[ms]" << std::endl;
        std::cout << "Pseudo GPU: Total decoding time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        std::cout << "Pseudo GPU decoding done!" << std::endl;
        std::cout << std::endl;
#endif

        printf("Pseudo GPU decoding done!\n");
        return BASE_SUCCESS;
    }

#ifdef INCLUDE_OPENCL

    std::size_t getLocalWorkSize(std::size_t globalWorkSize)
    {
        std::size_t localWorkSize = 1;
        while((localWorkSize < globalWorkSize) && (localWorkSize < 256)) {
            localWorkSize <<= 1;
        }
        return localWorkSize;
    }

    void getLocalAndGlobalWorkSize(std::size_t requiredThreads, std::size_t& localWorkSize, std::size_t& globalWorkSize)
    {
        localWorkSize  = getLocalWorkSize(requiredThreads);
        globalWorkSize = ((requiredThreads + localWorkSize - 1) / localWorkSize) * localWorkSize;
    }

    /**
 * @param width_a and @param height_a are related to channel size which is one half of the actual BayerCFA image.
 * BayerCFA image.
 */
    STATUS_t decodeBitstreamParallel_opencl(
       std::size_t width_a,
       std::size_t height_a,
       std::size_t lossyBits_a,
       std::size_t unaryMaxWidth_a,
       std::size_t bpp_a,
       const std::uint8_t* bitStream,
       const std::size_t bitStreamSize,
       std::uint8_t* bayerGB,
       std::size_t bayerGBSize,
       std::vector<std::uint32_t>& blockSizes)
    {

        std::size_t nrOfBlocks = blockSizes.size();
        printf("OpenCL decoding in blocks started!\n");
        identify_platforms();

        if(bpp_a != 8) {
            fprintf(
               stdout,
               "DecoderBase: bpp is not 8, but %zu. GPU decompression implemented only for 8 BPP.\n",
               bpp_a);
            return BASE_ERROR;
        }

        Reader reader{bitStream, bitStreamSize};

        std::size_t width;
        std::size_t height;
        std::size_t lossyBits;
        std::size_t unaryMaxWidth;
        unaryMaxWidth        = unaryMaxWidth_a;
        width                = width_a / 2;   // half size of the actual BayerCFA image
        height               = height_a / 2;
        lossyBits            = lossyBits_a;
        unaryMaxWidth        = unaryMaxWidth_a;
        std::uint32_t k_seed = bpp_a + 3;   // max 12 BPP + 3 = 15

        // Check if full decompressed image buffer is large enough. Multply by 4 because there are Gb,B,R & Gr channels.
        if(2 * width * 2 * height != bayerGBSize) {
            fprintf(
               stdout,
               "DecoderBase: expected size of output buffer: %zu, actual size: %zu (bpp: %zu)\n",
               2 * width * 2 * height,
               bayerGBSize,
               bpp_a);
            return BASE_OUTPUT_BUFFER_FALSE_SIZE;
        }

        // reader.loadFirstByte();
        reader.loadFirstFourBytes();

        //***************************************************
        // STEP 1: Discover and initialize the platforms
        //***************************************************

        cl_int status = 0;
        cl_uint numPlatforms;
        // cl_platform_id* platforms = NULL;

        // Get platforms
        status = clGetPlatformIDs(0, NULL, &numPlatforms);
        if(evaluateReturnStatus(status)) {
            return BASE_OPENCL_ERROR;
        };
        auto platforms = new cl_platform_id[numPlatforms];
        status         = clGetPlatformIDs(numPlatforms, platforms, NULL);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 2: Discover and initialize the devices
        //***************************************************

        cl_uint numDevices;
        // Use clGetDeviceIDs() to retrieve the number of
        // devices present
        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        evaluateReturnStatus(status);
        auto devices = new cl_device_id[numDevices];

        // Fill in devices with clGetDeviceIDs()
        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 3: Create a context
        //***************************************************

        cl_context context = NULL;
        // Create a context using clCreateContext() and
        // associate it with the devices
        context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 4: Create a command queue
        //***************************************************
        cl_command_queue cmdQueue;
        // Create a command queue using clCreateCommandQueue(),
        // and associate it with the device you want to execute
        // on
        // Command queue properties
        cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cmdQueue = clCreateCommandQueueWithProperties(context, devices[0], queue_properties, &status);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 5: Create device buffers
        //***************************************************

        cl_mem YCCC_d;   // output buffer
        cl_mem YCCC_dpcm_d;   // input buffer
        cl_mem bayerGB_d;   // output buffer
        size_t iNumElements         = 4 * (height * width);
        size_t datasize_YCCC_d      = sizeof(std::int16_t) * iNumElements;
        size_t datasize_YCCC_dpcm_d = sizeof(std::int16_t) * iNumElements;
        size_t datasize_BayerGB     = 2 * width * 2 * height;
        YCCC_d                      = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_YCCC_d, NULL, &status);
        evaluateReturnStatus(status);
        YCCC_dpcm_d = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_YCCC_dpcm_d, NULL, &status);
        evaluateReturnStatus(status);
        bayerGB_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize_BayerGB, NULL, &status);

        //***************************************************
        // STEP 6: Create a program object for the context
        //***************************************************

        FILE* programHandle;   // File that contains kernel functions
        size_t programSize;
        char* programBuffer;
        cl_program cpProgram;

        std::filesystem::path cwd = std::filesystem::current_path();
        std::cout << "Current working dir: " << cwd << std::endl;

        // Construct the full path to the kernel file
        std::filesystem::path kernelFilePath = cwd / "OpenCL_sources/kernels.cl";
        std::cout << "Kernel file will be searched for at: " << kernelFilePath << std::endl;

        programHandle = fopen(kernelFilePath.string().c_str(), "rb");
        // programHandle = fopen("OpenCL_sources/kernel_dpcm_row.cl", "r");
        fseek(programHandle, 0, SEEK_END);
        programSize = ftell(programHandle);
        rewind(programHandle);

        printf("Program size = %llu B \n", programSize);

        // 6 b: read the kernel source into the buffer programBuffer
        //      add null-termination-required by clCreateProgramWithSource
        programBuffer = (char*)malloc(programSize + 1);

        programBuffer[programSize] = '\0';   // add null-termination
        fread(programBuffer, sizeof(char), programSize, programHandle);
        fclose(programHandle);

        // 6 c: Create the program from the source
        //
        cpProgram = clCreateProgramWithSource(context, 1, (const char**)&programBuffer, &programSize, &status);
        evaluateReturnStatus(status);
        free(programBuffer);
        //***************************************************
        // STEP 7: Build the program
        //***************************************************

        status = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);

        if(status != CL_SUCCESS) {
            size_t len;

            printf("Error: Failed to build program executable!\n");
            // Firstly, get the length of the error message:
            status = clGetProgramBuildInfo(cpProgram, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            evaluateReturnStatus(status);

            // allocate enough memory to store the error message:
            char* err_buffer = (char*)malloc(len * sizeof(char));

            // Secondly, copy the error message into buffer
            status =
               clGetProgramBuildInfo(cpProgram, devices[0], CL_PROGRAM_BUILD_LOG, len * sizeof(char), err_buffer, NULL);
            evaluateReturnStatus(status);
            printf("%s\n", err_buffer);
            free(err_buffer);
            exit(1);
        }

        //***************************************************
        // STEP 8: Create and compile the kernel
        //***************************************************

        cl_kernel ckBitstreamToDpcm;
        ckBitstreamToDpcm = clCreateKernel(cpProgram, "bitstream_to_dpcm", &status);
        evaluateReturnStatus(status);

        cl_kernel ckFirstColumnAllRows;
        ckFirstColumnAllRows = clCreateKernel(cpProgram, "first_column_all_rows", &status);
        evaluateReturnStatus(status);

        cl_kernel ckDpcmAcrossRows;
        ckDpcmAcrossRows = clCreateKernel(cpProgram, "dpcm_across_rows", &status);
        evaluateReturnStatus(status);

        cl_kernel ckYcccToBayerGB;
        ckYcccToBayerGB = clCreateKernel(cpProgram, "yccc_to_bayergb_8bit", &status);
        evaluateReturnStatus(status);

        //***************************************************
        // OPENCL NOW READY TO EXECUTE
        //***************************************************

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#    endif

        //***************************************************
        // STEP 8a: Set up kernel for bitstream to DPCM decoding
        //***************************************************

        //         // write bitstream to global GPU memory
        //         status = clEnqueueWriteBuffer(cmdQueue, bitStream_d, CL_FALSE, 0, bitStreamSize, bitStream, 0, NULL, NULL);
        //         evaluateReturnStatus(status);

        std::size_t rowsPerBlock          = (height + (nrOfBlocks - 1)) / nrOfBlocks;
        std::size_t blockSize_quadruplets = rowsPerBlock * width;
        std::size_t blockSize_pixels      = 4 * blockSize_quadruplets;

        std::uint32_t pixel_current = 0;
        std::uint32_t pixel_all     = 2 * width * 2 * height;

        std::vector<std::uint32_t> pixelsInBlock(nrOfBlocks, 0);

        for(std::size_t i = 0; i < nrOfBlocks; i++) {
            pixel_current += blockSize_pixels;
            if(pixel_current >= pixel_all) {
                pixelsInBlock[i] = blockSize_pixels - pixel_current % pixel_all;
                break;
            } else {
                pixelsInBlock[i] = blockSize_pixels;
            }
        }

        // printf("OpenCL: Block size in pixels:\n");
        // for(std::size_t i = 0; i < nrOfBlocks; i++) {
        //     printf("Block %zu: %u\n", i, pixelsInBlock[i]);
        // }

        std::uint64_t groupByteOffset = (((iNumElements / nrOfBlocks * (unaryMaxWidth + k_seed) + 7) / 8));
        std::uint64_t requiredSpace   = groupByteOffset * nrOfBlocks;

        printf("OpenCL: Nr of blocks: %zu\n", nrOfBlocks);
        printf("OpenCL: Group byte offset: %llu B\n", groupByteOffset);
        printf("OpenCL: Required space for bitstream: %llu B\n", requiredSpace);

        cl_mem bitStream_d;   // input bitstream buffer
        bitStream_d = clCreateBuffer(context, CL_MEM_READ_ONLY, requiredSpace, NULL, &status);
        evaluateReturnStatus(status);

        // create on device buffer for pixelsInBlock array
        cl_mem pixelsInBlock_d;
        pixelsInBlock_d =
           clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(std::uint32_t) * pixelsInBlock.size(), NULL, &status);
        evaluateReturnStatus(status);

        printf("OpenCL: Bitstream transfer to GPU memory \n");
#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_bitstream_to_dpcm_memory = std::chrono::steady_clock::now();
#    endif

        // BITSTREAM DATA TRANSFER
        std::size_t host_offset   = 0;
        std::size_t device_offset = 0;
        std::size_t b             = 0;
        for(auto size : blockSizes) {
            // printf(
            //    "OpenCL: Bitstream transfer: Block %zu | Host offset: %zu, Device offset: %zu, Size: %u",
            //    b,
            //    host_offset,
            //    device_offset,
            //    size);
            // if(nrOfBlocks < 256) {
            //     printf("\n");
            // }else{
            //     printf("\r");
            // }
            status = clEnqueueWriteBuffer(
               cmdQueue,
               bitStream_d,
               CL_FALSE,
               device_offset,
               size,
               &(bitStream[host_offset]),
               0,
               NULL,
               NULL);
            evaluateReturnStatus(status);
            host_offset += size;
            device_offset += groupByteOffset;
            b++;
        }
        printf("\n");

        // PIXELS IN BLOCK TRANSFER
        status = clEnqueueWriteBuffer(
           cmdQueue,
           pixelsInBlock_d,
           CL_FALSE,
           0,
           sizeof(std::uint32_t) * pixelsInBlock.size(),
           pixelsInBlock.data(),
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_bitstream_to_dpcm_memory = std::chrono::steady_clock::now();
#    endif

        std::int32_t nrOfRowsInBlock = (height + (nrOfBlocks - 1)) / nrOfBlocks;

        printf("OpenCL: Nr of rows in block: %d\n", nrOfRowsInBlock);

        // Set the Argument values
        status = clSetKernelArg(ckBitstreamToDpcm, 0, sizeof(cl_mem), (void*)&bitStream_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 1, sizeof(cl_mem), (void*)&pixelsInBlock_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 2, sizeof(cl_mem), (void*)&YCCC_dpcm_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 3, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 4, sizeof(cl_uint), (void*)&unaryMaxWidth);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 5, sizeof(cl_uint), (void*)&bpp_a);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 6, sizeof(cl_ulong), (void*)&groupByteOffset);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 7, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 8, sizeof(cl_int), (void*)&height);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckBitstreamToDpcm, 9, sizeof(cl_int), (void*)&nrOfRowsInBlock);
        evaluateReturnStatus(status);

        size_t bitstreamToDpcm_globalSize;   // = nrOfBlocks;
        size_t bitstreamToDpcm_localSize;   //  = getLocalWorkSize(nrOfBlocks);

        getLocalAndGlobalWorkSize(nrOfBlocks, bitstreamToDpcm_localSize, bitstreamToDpcm_globalSize);

        printf("OpenCL: Bitstream to DPCM kernel: Global work size: %zu\n", bitstreamToDpcm_globalSize);
        printf("OpenCL: Bitstream to DPCM kernel:  Local work size: %zu\n", bitstreamToDpcm_localSize);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_parsing = std::chrono::steady_clock::now();
#    endif

        // Execute the kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckBitstreamToDpcm,
           1,
           NULL,
           &bitstreamToDpcm_globalSize,
           &bitstreamToDpcm_localSize,
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

        printf("OpenCL: Bitstream to DPCM kernel executed\n");

        clFinish(cmdQueue);

        printf("OpenCL: Bitstream to DPCM kernel finished\n");

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_parsing = std::chrono::steady_clock::now();
#    endif

        //***************************************************
        // STEP 8b: Set up kernel for calculation of first column values for each block
        //***************************************************

        status = clSetKernelArg(ckFirstColumnAllRows, 0, sizeof(cl_mem), (void*)&YCCC_dpcm_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckFirstColumnAllRows, 1, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckFirstColumnAllRows, 2, sizeof(cl_mem), (void*)&pixelsInBlock_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckFirstColumnAllRows, 3, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckFirstColumnAllRows, 4, sizeof(cl_int), (void*)&height);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckFirstColumnAllRows, 5, sizeof(cl_int), (void*)&nrOfRowsInBlock);
        evaluateReturnStatus(status);

        size_t firstColumnAllRows_globalSize;
        size_t firstColumnAllRows_localSize;

        getLocalAndGlobalWorkSize(nrOfBlocks, firstColumnAllRows_localSize, firstColumnAllRows_globalSize);

        printf("OpenCL: First column all rows kernel: Global work size: %zu\n", firstColumnAllRows_globalSize);
        printf("OpenCL: First column all rows kernel:  Local work size: %zu\n", firstColumnAllRows_localSize);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_firstColumnAllRows = std::chrono::steady_clock::now();
#    endif

        // Execute the kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckFirstColumnAllRows,
           1,
           NULL,
           &firstColumnAllRows_globalSize,
           &firstColumnAllRows_localSize,
           0,
           NULL,
           NULL);

        evaluateReturnStatus(status);

        printf("OpenCL: First column all rows kernel executed\n");

        clFinish(cmdQueue);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_firstColumnAllRows = std::chrono::steady_clock::now();
#    endif

        printf("OpenCL: First column all rows kernel finished\n");

        //***************************************************
        // STEP 10: Set the kernel arguments for dpcm to full YCCC calculation
        //***************************************************

        // Set the Argument values
        status = clSetKernelArg(ckDpcmAcrossRows, 0, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 1, sizeof(cl_mem), (void*)&YCCC_dpcm_d);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 2, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 3, sizeof(cl_int), (void*)&height);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 11: Enqueue the kernel for execution
        //***************************************************

        //***************************************************
        // STEP 2.5: Calculate work sizes
        //***************************************************

        // number of work-items in the work-group; defines overall size of the N-Dimensional range
        size_t szGlobalWorkSize = height;
        // 128 threads per work group, must divide evenly into global work size
        size_t szLocalWorkSize;
        clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(szLocalWorkSize), &szLocalWorkSize, NULL);
        // Adjust global work size to be a multiple of local work size
        if(szGlobalWorkSize % szLocalWorkSize != 0) {
            szGlobalWorkSize = ((szGlobalWorkSize / szLocalWorkSize) + 1) * szLocalWorkSize;
        }
        printf(
           "OpenCL dpcm_to_yccc kernel: Global work size: %zu, Local work size: %zu\n",
           szGlobalWorkSize,
           szLocalWorkSize);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_dpcm_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        // Launch kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckDpcmAcrossRows,
           1,
           NULL,
           &szGlobalWorkSize,
           &szLocalWorkSize,
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_dpcm_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        //***************************************************
        // STEP 12: Read the output buffer back to the host
        //***************************************************
        // Synchronous/blocking read of results
        // Read back full YCCC values
        // TODO: remove this because it is not actually necesarry
        // status = clEnqueueReadBuffer(cmdQueue, YCCC_d, CL_TRUE, 0, datasize_YCCC_d, YCCC_h.data(), 0, NULL, NULL);
        // evaluateReturnStatus(status);

        //***************************************************
        // STEP 13: Set the kernel arguments for yccc to bayerGB conversion
        //***************************************************

        status = clSetKernelArg(ckYcccToBayerGB, 0, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 1, sizeof(cl_mem), (void*)&bayerGB_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 2, sizeof(cl_int), (void*)&lossyBits);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 3, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        cl_int nrOfPixels = height * width;
        status            = clSetKernelArg(ckYcccToBayerGB, 4, sizeof(cl_int), (void*)&nrOfPixels);
        evaluateReturnStatus(status);

        //***************************************************

        std::size_t szGlobalWorkSize_yccc_to_bayer = height * width;
        std::size_t szLocalWorkSize_yccc_to_bayer;
        clGetDeviceInfo(
           devices[0],
           CL_DEVICE_MAX_WORK_GROUP_SIZE,
           sizeof(szLocalWorkSize_yccc_to_bayer),
           &szLocalWorkSize_yccc_to_bayer,
           NULL);
        if(szGlobalWorkSize_yccc_to_bayer % szLocalWorkSize_yccc_to_bayer != 0) {
            szGlobalWorkSize_yccc_to_bayer =
               ((szGlobalWorkSize_yccc_to_bayer / szLocalWorkSize_yccc_to_bayer) + 1) * szLocalWorkSize_yccc_to_bayer;
        }
        printf(
           "OpenCL yccc_to_bayer kernel: Global work size: %zu, Local work size: %zu\n",
           szGlobalWorkSize_yccc_to_bayer,
           szLocalWorkSize_yccc_to_bayer);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_yccc_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        // Launch kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckYcccToBayerGB,
           1,
           NULL,
           &szGlobalWorkSize_yccc_to_bayer,
           &szLocalWorkSize_yccc_to_bayer,
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_yccc_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_yccc_2_bayer_memory = std::chrono::steady_clock::now();
#    endif
        // Read back BayerGB values
        status = clEnqueueReadBuffer(cmdQueue, bayerGB_d, CL_TRUE, 0, datasize_BayerGB, bayerGB, 0, NULL, NULL);
        evaluateReturnStatus(status);
        // Block until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed
#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_yccc_2_bayer_memory = std::chrono::steady_clock::now();
#    endif

        printf("GPU execution finished\n");

        // char outputFile[200];
        // char txt[200];
        // sprintf(outputFile, "C:/DATA/Repos/OMLS_Masters_SW/images/lite_dataset/dump/gpu_DPCM.txt");
        // std::ofstream wf(outputFile, std::ios::out);
        // if(!wf) {
        //     char msg[200];
        //     sprintf(msg, "Cannot open specified file: %s", outputFile);
        //     throw std::runtime_error(msg);
        // }
        // std::sprintf(txt, "%zu %zu pred YCCC, curr YCCC, dpcm\n", width, height);
        // wf << txt;
        // int16_t mask_Y      = (1 << (bpp_a + 2)) - 1;   // 0x03FF
        // int16_t mask_C      = (1 << (bpp_a + 1)) - 1;   // 0x01FF;
        // int16_t mask_Y_dpcm = (1 << (bpp_a + 3)) - 1;   // 0x07FF;
        // int16_t mask_C_dpcm = (1 << (bpp_a + 2)) - 1;   // 0x03FF;

        // for(std::size_t i = 0; i < width * height; i++) {
        //     std::sprintf(
        //        txt,
        //        "%4zu %4zu : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX\n",
        //        i / width,
        //        i % width,
        //        (YCCC_prev_debug[4 * i + 0] & mask_Y),
        //        (YCCC_prev_debug[4 * i + 1] & mask_C),
        //        (YCCC_prev_debug[4 * i + 2] & mask_C),
        //        (YCCC_prev_debug[4 * i + 3] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 0] & mask_Y),
        //        //    (YCCC_h_gt[4 * i + 1] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 2] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 3] & mask_C),
        //        (YCCC_h[4 * i + 0] & mask_Y),
        //        (YCCC_h[4 * i + 1] & mask_C),
        //        (YCCC_h[4 * i + 2] & mask_C),
        //        (YCCC_h[4 * i + 3] & mask_C),
        //        (YCCC_dpcm_h[4 * i + 0] & mask_Y_dpcm),
        //        (YCCC_dpcm_h[4 * i + 1] & mask_C_dpcm),
        //        (YCCC_dpcm_h[4 * i + 2] & mask_C_dpcm),
        //        (YCCC_dpcm_h[4 * i + 3] & mask_C_dpcm));
        //     wf << txt;
        // }
        // wf.close();

        //***************************************************
        // STEP 13: Release OpenCL resources
        //***************************************************
        if(ckBitstreamToDpcm)
            clReleaseKernel(ckBitstreamToDpcm);
        if(bitStream_d)
            clReleaseMemObject(bitStream_d);
        if(pixelsInBlock_d)
            clReleaseMemObject(pixelsInBlock_d);

        if(ckFirstColumnAllRows)
            clReleaseKernel(ckFirstColumnAllRows);


        if(ckDpcmAcrossRows)
            clReleaseKernel(ckDpcmAcrossRows);
        if(YCCC_dpcm_d)
            clReleaseMemObject(YCCC_dpcm_d);
        if(YCCC_d)
            clReleaseMemObject(YCCC_d);

        if(ckYcccToBayerGB)
            clReleaseKernel(ckYcccToBayerGB);
        if(bayerGB_d)
            clReleaseMemObject(bayerGB_d);


        if(cpProgram)
            clReleaseProgram(cpProgram);
        if(cmdQueue)
            clReleaseCommandQueue(cmdQueue);
        if(context)
            clReleaseContext(context);

        delete[] platforms;
        delete[] devices;

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl;
        std::cout << "OpenCL: Parallel decoding time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        std::cout << "OpenCL: Memory transfer Bitstream to DPCM time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_bitstream_to_dpcm_memory - begin_bitstream_to_dpcm_memory)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Parsing time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parsing - begin_parsing).count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Kernel first column all rows time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_firstColumnAllRows - begin_firstColumnAllRows)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Kernel DPCM to YCCC time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_dpcm_2_bayer_kernel - begin_dpcm_2_bayer_kernel)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Kernel YCCC to BayerGB time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_yccc_2_bayer_kernel - begin_yccc_2_bayer_kernel)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Memory transfer YCCC to BayerGB time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_yccc_2_bayer_memory - begin_yccc_2_bayer_memory)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << std::endl;

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        printf(
           "Resolution: %zu x %zu | Pixels: %zu |Nr of blocks: %zu | Mem Bitstream: %zu ms | Parsing time: %zu ms | "
           "First column: %zu ms | DPCM to YCCC: %zu ms | YCCC to BayerGB: %zu ms | Mem transfer device host: %zu ms | "
           "Total time: %zu ms\n",
           2 * width,
           2 * height,
           2 * width * 2 * height,
           nrOfBlocks,
           std::chrono::duration_cast<std::chrono::milliseconds>(
              end_bitstream_to_dpcm_memory - begin_bitstream_to_dpcm_memory)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_parsing - begin_parsing).count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_firstColumnAllRows - begin_firstColumnAllRows)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_dpcm_2_bayer_kernel - begin_dpcm_2_bayer_kernel)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_yccc_2_bayer_kernel - begin_yccc_2_bayer_kernel)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_yccc_2_bayer_memory - begin_yccc_2_bayer_memory)
              .count(),
           total_time);

        printf(
           "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
           2 * width,
           2 * height,
           2 * width * 2 * height,
           nrOfBlocks,
           std::chrono::duration_cast<std::chrono::milliseconds>(
              end_bitstream_to_dpcm_memory - begin_bitstream_to_dpcm_memory)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_parsing - begin_parsing).count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_firstColumnAllRows - begin_firstColumnAllRows)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_dpcm_2_bayer_kernel - begin_dpcm_2_bayer_kernel)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_yccc_2_bayer_kernel - begin_yccc_2_bayer_kernel)
              .count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_yccc_2_bayer_memory - begin_yccc_2_bayer_memory)
              .count(),
           total_time);

#    endif

        printf("OpenCL: Parallel decoding finished\n");
        return BASE_SUCCESS;
    }

    /*********************
 * Implementatiton for no blocks.
 * @param width_a and @param height_a are related to channel size which is one half of the actual BayerCFA image.
 * BayerCFA image.
 */
    STATUS_t decodeBitstreamParallel_opencl(
       std::size_t width_a,
       std::size_t height_a,
       std::size_t lossyBits_a,
       std::size_t unaryMaxWidth_a,
       std::size_t bpp_a,
       const std::uint8_t* bitStream,
       const std::size_t bitStreamSize,
       std::uint8_t* bayerGB,
       std::size_t bayerGBSize)
    {

        printf("OpenCL decoding started!\n");
        // identify_platforms();

        if(bpp_a != 8) {
            fprintf(
               stdout,
               "DecoderBase: bpp is not 8, but %zu. GPU decompression implemented only for 8 BPP.\n",
               bpp_a);
            return BASE_ERROR;
        }

        std::uint32_t N_threshold = 8;
        std::uint32_t A_init      = 32;

        Reader reader{bitStream, bitStreamSize};

        std::size_t width;
        std::size_t height;
        std::size_t lossyBits;
        std::size_t unaryMaxWidth;
        unaryMaxWidth        = unaryMaxWidth_a;
        width                = width_a / 2;   // half size of the actual BayerCFA image
        height               = height_a / 2;
        lossyBits            = lossyBits_a;
        unaryMaxWidth        = unaryMaxWidth_a;
        std::uint32_t k_seed = bpp_a + 3;   // max 12 BPP + 3 = 15

        // Check if full decompressed image buffer is large enough. Multply by 4 because there are Gb,B,R & Gr channels.
        if(2 * width * 2 * height != bayerGBSize) {
            fprintf(
               stdout,
               "DecoderBase: expected size of output buffer: %zu, actual size: %zu (bpp: %zu)\n",
               2 * width * 2 * height,
               bayerGBSize,
               bpp_a);
            return BASE_OUTPUT_BUFFER_FALSE_SIZE;
        }

        // reader.loadFirstByte();
        reader.loadFirstFourBytes();

        //***************************************************
        // STEP 1: Discover and initialize the platforms
        //***************************************************

        cl_int status = 0;
        cl_uint numPlatforms;
        // cl_platform_id* platforms = NULL;

        // Get platforms
        status = clGetPlatformIDs(0, NULL, &numPlatforms);
        if(evaluateReturnStatus(status)) {
            return BASE_OPENCL_ERROR;
        };
        auto platforms = new cl_platform_id[numPlatforms];
        status         = clGetPlatformIDs(numPlatforms, platforms, NULL);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 2: Discover and initialize the devices
        //***************************************************

        cl_uint numDevices;
        // Use clGetDeviceIDs() to retrieve the number of
        // devices present
        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        evaluateReturnStatus(status);
        auto devices = new cl_device_id[numDevices];

        // Fill in devices with clGetDeviceIDs()
        status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 2.5: Calculate work sizes
        //***************************************************

        // number of work-items in the work-group; defines overall size of the N-Dimensional range
        size_t szGlobalWorkSize = height;
        // 128 threads per work group, must divide evenly into global work size
        size_t szLocalWorkSize;
        clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(szLocalWorkSize), &szLocalWorkSize, NULL);
        // Adjust global work size to be a multiple of local work size
        if(szGlobalWorkSize % szLocalWorkSize != 0) {
            szGlobalWorkSize = ((szGlobalWorkSize / szLocalWorkSize) + 1) * szLocalWorkSize;
        }
        printf(
           "OpenCL dpcm_to_yccc kernel: Global work size: %zu, Local work size: %zu\n",
           szGlobalWorkSize,
           szLocalWorkSize);

        std::size_t szGlobalWorkSize_yccc_to_bayer = height * width;
        std::size_t szLocalWorkSize_yccc_to_bayer;
        clGetDeviceInfo(
           devices[0],
           CL_DEVICE_MAX_WORK_GROUP_SIZE,
           sizeof(szLocalWorkSize_yccc_to_bayer),
           &szLocalWorkSize_yccc_to_bayer,
           NULL);
        if(szGlobalWorkSize_yccc_to_bayer % szLocalWorkSize_yccc_to_bayer != 0) {
            szGlobalWorkSize_yccc_to_bayer =
               ((szGlobalWorkSize_yccc_to_bayer / szLocalWorkSize_yccc_to_bayer) + 1) * szLocalWorkSize_yccc_to_bayer;
        }
        printf(
           "OpenCL yccc_to_bayer kernel: Global work size: %zu, Local work size: %zu\n",
           szGlobalWorkSize_yccc_to_bayer,
           szLocalWorkSize_yccc_to_bayer);

        //***************************************************
        // STEP 3: Create a context
        //***************************************************

        cl_context context = NULL;
        // Create a context using clCreateContext() and
        // associate it with the devices
        context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 4: Create a command queue
        //***************************************************
        cl_command_queue cmdQueue;
        // Create a command queue using clCreateCommandQueue(),
        // and associate it with the device you want to execute
        // on
        // Command queue properties
        cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cmdQueue = clCreateCommandQueueWithProperties(context, devices[0], queue_properties, &status);
        evaluateReturnStatus(status);

        //***************************************************
        // STEP 5: Create device buffers
        //***************************************************
        cl_mem YCCC_d;   // output buffer
        cl_mem YCCC_dpcm_d;   // input buffer
        cl_mem bayerGB_d;   // output buffer
        size_t iNumElements         = 4 * (height * width);
        size_t datasize_YCCC_d      = sizeof(std::int16_t) * iNumElements;
        size_t datasize_YCCC_dpcm_d = sizeof(std::int16_t) * iNumElements;
        size_t datasize_BayerGB     = 2 * width * 2 * height;
        YCCC_d                      = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_YCCC_d, NULL, &status);
        evaluateReturnStatus(status);
        YCCC_dpcm_d = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_YCCC_dpcm_d, NULL, &status);
        evaluateReturnStatus(status);
        bayerGB_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize_BayerGB, NULL, &status);

        //***************************************************
        // STEP 6: Create a program object for the context
        //***************************************************

        FILE* programHandle;   // File that contains kernel functions
        size_t programSize;
        char* programBuffer;
        cl_program cpProgram;

        std::filesystem::path cwd = std::filesystem::current_path();
        std::cout << "Current working dir: " << cwd << std::endl;

        // Construct the full path to the kernel file
        std::filesystem::path kernelFilePath = cwd / "OpenCL_sources/kernels.cl";
        std::cout << "Kernel file will be searched for at: " << kernelFilePath << std::endl;

        programHandle = fopen(kernelFilePath.string().c_str(), "rb");
        // programHandle = fopen("OpenCL_sources/kernel_dpcm_row.cl", "r");
        fseek(programHandle, 0, SEEK_END);
        programSize = ftell(programHandle);
        rewind(programHandle);

        printf("Program size = %llu B \n", programSize);

        // 6 b: read the kernel source into the buffer programBuffer
        //      add null-termination-required by clCreateProgramWithSource
        programBuffer = (char*)malloc(programSize + 1);

        programBuffer[programSize] = '\0';   // add null-termination
        fread(programBuffer, sizeof(char), programSize, programHandle);
        fclose(programHandle);

        // 6 c: Create the program from the source
        //
        cpProgram = clCreateProgramWithSource(context, 1, (const char**)&programBuffer, &programSize, &status);
        evaluateReturnStatus(status);
        free(programBuffer);
        //***************************************************
        // STEP 7: Build the program
        //***************************************************

        status = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);

        if(status != CL_SUCCESS) {
            size_t len;

            printf("Error: Failed to build program executable!\n");
            // Firstly, get the length of the error message:
            status = clGetProgramBuildInfo(cpProgram, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            evaluateReturnStatus(status);

            // allocate enough memory to store the error message:
            char* err_buffer = (char*)malloc(len * sizeof(char));

            // Secondly, copy the error message into buffer
            status =
               clGetProgramBuildInfo(cpProgram, devices[0], CL_PROGRAM_BUILD_LOG, len * sizeof(char), err_buffer, NULL);
            evaluateReturnStatus(status);
            printf("%s\n", err_buffer);
            free(err_buffer);
            exit(1);
        }

        //***************************************************
        // STEP 8: Create and compile the kernel
        //***************************************************

        cl_kernel ckDpcmAcrossRows;
        // Create the kernel
        ckDpcmAcrossRows = clCreateKernel(cpProgram, "dpcm_across_rows", &status);
        evaluateReturnStatus(status);
        cl_kernel ckYcccToBayerGB;
        ckYcccToBayerGB = clCreateKernel(cpProgram, "yccc_to_bayergb_8bit", &status);
        evaluateReturnStatus(status);

        //***************************************************
        // OPENCL NOW READY TO EXECUTE
        //***************************************************

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin         = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point begin_parsing = std::chrono::steady_clock::now();
#    endif

        std::uint32_t A[] = {A_init, A_init, A_init, A_init};
        std::uint32_t N   = N_START;
        std::vector<std::int16_t> YCCC_h(4 * (height * width), 255);
        std::vector<std::int16_t> YCCC_prev_debug(4 * (height * width));
        std::int16_t YCCC_prev[] = {0, 0, 0, 0};
        std::vector<std::int16_t> YCCC_dpcm_h(4 * (height * width));

        // 1.) decode 4 seed pixels
        // Seed pixel
        std::uint16_t posValue[] = {0, 0, 0, 0};
        for(std::size_t ch = 0; ch < 4; ch++) {
            // for(std::uint32_t n = k_SEED + 1; n > 0; n--) { // MSB first
            //     std::uint32_t lastBit = fetchBit(reader);
            //     posValue[ch]          = (posValue[ch] << 1) | lastBit;
            // }
            (void)reader.fetchBit_buffered();   // read delimiter
            for(std::uint32_t n = 0; n < k_seed; n++) {   // LSB first
                std::uint32_t lastBit = reader.fetchBit_buffered();
                posValue[ch]          = posValue[ch] | ((std::uint16_t)lastBit << n);
            };
            YCCC_h[0 + ch] = DecoderBase::fromAbs(posValue[ch]);
        }

        for(std::size_t ch = 0; ch < 4; ch++) {   //
            A[ch] += YCCC_h[0 + ch] > 0 ? YCCC_h[0 + ch] : -YCCC_h[0 + ch];
        }

        memcpy(YCCC_prev, YCCC_h.data(), sizeof(YCCC_prev));

        YCCC_dpcm_h[0] = YCCC_h[0];
        YCCC_dpcm_h[1] = YCCC_h[1];
        YCCC_dpcm_h[2] = YCCC_h[2];
        YCCC_dpcm_h[3] = YCCC_h[3];

#    ifdef USE_LUT

        std::uint8_t offset_i = 0;
        std::uint8_t address  = 0;
        // Get DPCM from binary bitstream (reverse Golomb Rice)
        for(std::size_t idx = 1; idx < height * width; idx++) {

            // 2.) AGOR
            std::uint16_t quotient[]  = {0, 0, 0, 0};
            std::uint16_t remainder[] = {0, 0, 0, 0};
            std::uint16_t k[]         = {0, 0, 0, 0};
            //std::int16_t dpcm_curr[]  = {0, 0, 0, 0};

            for(std::size_t ch = 0; ch < 4; ch++) {
                for(std::size_t it = 0; it < (bpp_a + 2); it++) {
                    if((N << k[ch]) < A[ch]) {
                        k[ch] = k[ch] + 1;
                    } else {
                        k[ch] = k[ch];
                    }
                }
                std::uint32_t lastBit;

                std::uint16_t absVal = 0;
                do {   // decode quotient: unary coding
                    lastBit = reader.fetchBit_buffered();
                    if(lastBit == 1) {   //
                        quotient[ch]++;
                    }
                } while(lastBit == 1 && quotient[ch] < unaryMaxWidth);

                lut_data_t lut_data;
                lut(bitStream[address], offset_i, k[ch], &lut_data);

                if(lut_data.q_done == false) {
                    quotient[ch] = lut_data.quotient;
                    address++;
                    // offset_i = lut_data.offset_o;
                    auto quotient_i = quotient[ch];
                    lut(bitStream[address], 0, k[ch], quotient_i, &lut_data);
                    quotient[ch] += lut_data.quotient;
                    if(quotient[ch] >= unaryMaxWidth) {
                        absVal = lut_data.truncated_coding_value << lut_data.t_count;
                        while(lut_data.t_count > 0) {
                            address++;
                            r_lut(bitStream[address], lut_data.t_count, &lut_data);
                            absVal += lut_data.remainder << lut_data.t_count;
                        }
                    } else {
                        remainder[ch] = lut_data.remainder << lut_data.r_count;
                        while(lut_data.r_count > 0) {
                            address++;
                            r_lut(bitStream[address], lut_data.r_count, &lut_data);
                            remainder[ch] += lut_data.remainder << lut_data.r_count;
                        }

                        absVal = quotient[ch] << k[ch] + remainder[ch];
                    }
                    offset_i = lut_data.offset_o;

                    // else if(lut_data.quotient >= unaryMaxWidth) {   // whole byte are 1s
                    //     address++;
                    //     r_lut(bitStream[address], k_seed, &lut_data);
                    //     absVal = lut_data.remainder << lut_data.r_count;

                    //     // address++;
                    //     // r_lut(
                    //     //    bitStream[address],
                    //     //    lut_data.r_count,
                    //     //    &lut_data);   // this only work for k_seed <= 16; otherwise make a while loop to iterate as long as r_count > 0
                    //     // absVal += lut_data.remainder;
                    //     while(lut_data.t_count > 0) {
                    //         address++;
                    //         r_lut(bitStream[address], lut_data.t_count, &lut_data);
                    //         absVal += lut_data.remainder << lut_data.t_count;
                    //     }
                    //     offset_i = lut_data.offset_o;
                } else {
                    remainder[ch] = lut_data.remainder << lut_data.r_count;
                    while(lut_data.r_count > 0) {
                        address++;
                        r_lut(bitStream[address], lut_data.r_count, &lut_data);
                        remainder[ch] += lut_data.remainder << lut_data.r_count;
                    }
                    absVal = quotient[ch] << k[ch] + remainder[ch];
                }

                /* m_unaryMaxWidth * '1' -> indicates binary coding of positive value */
                if(quotient[ch] >= unaryMaxWidth) {
                    // for(std::uint32_t n = 0; n < k_seed; n++) {   //decode remainder
                    //     lastBit = reader.fetchBit_buffered();
                    //     absVal  = absVal | ((std::uint16_t)lastBit << n); /* LSB first */
                    //     // absVal  = (absVal << 1) | lastBit; /* MSB first*/
                    // };
                    // absVal = reader.getMSBfirst(reader.fetchBits_buffered(k_seed), k_seed);
                    absVal = getMSBfirst(reader.fetchBits_buffered(k_seed), k_seed);
                    // absVal = reader.fetchBits_buffered(k_seed);
                    // auto temp = absVal;
                } else {
                    // if(k[ch] == 0) {
                    //     absVal = quotient[ch];
                    // } else {
                    // for(std::uint32_t n = 0; n < k[ch]; n++) {   //decode remainder
                    //     lastBit       = reader.fetchBit_buffered();
                    //     remainder[ch] = remainder[ch] | ((std::uint16_t)lastBit << n); /* LSB first*/
                    //     // remainder[ch] = (remainder[ch] << 1) | lastBit; /* MSB first*/
                    // };
                    // remainder[ch] = (uint16_t)reader.fetchBits_buffered(k[ch]);
                    // remainder[ch] = (uint16_t)reader.getMSBfirst(reader.fetchBits_buffered(k[ch]), k[ch]);
                    remainder[ch] = (uint16_t)getMSBfirst(reader.fetchBits_buffered(k[ch]), k[ch]);
                    // auto temp = remainder[ch];
                    // auto temp_2 = reader.getMSBfirst(temp, k[ch]);
                    // absVal = quotient[ch] * (1 << k[ch]) + remainder[ch];
                    absVal = (quotient[ch] << k[ch]) + remainder[ch];
                    // }
                }

                // TODO: OPPORTUNITY: unsigned absVal to signed dpcm can also be done in parallel
                YCCC_dpcm_h[4 * idx + ch] = DecoderBase::fromAbs(absVal);
            }

            A[0] += YCCC_dpcm_h[4 * idx + 0] > 0 ? YCCC_dpcm_h[4 * idx + 0] : -YCCC_dpcm_h[4 * idx + 0];
            A[1] += YCCC_dpcm_h[4 * idx + 1] > 0 ? YCCC_dpcm_h[4 * idx + 1] : -YCCC_dpcm_h[4 * idx + 1];
            A[2] += YCCC_dpcm_h[4 * idx + 2] > 0 ? YCCC_dpcm_h[4 * idx + 2] : -YCCC_dpcm_h[4 * idx + 2];
            A[3] += YCCC_dpcm_h[4 * idx + 3] > 0 ? YCCC_dpcm_h[4 * idx + 3] : -YCCC_dpcm_h[4 * idx + 3];

            N += 1;
            if(N >= N_threshold) {
                N >>= 1;
                A[0] >>= 1;
                A[1] >>= 1;
                A[2] >>= 1;
                A[3] >>= 1;
            }
            A[0] = A[0] < A_MIN ? A_MIN : A[0];
            A[1] = A[1] < A_MIN ? A_MIN : A[1];
            A[2] = A[2] < A_MIN ? A_MIN : A[2];
            A[3] = A[3] < A_MIN ? A_MIN : A[3];
        }

#    else
        // Get DPCM from binary bitstream (reverse Golomb Rice)
        for(std::size_t idx = 1; idx < height * width; idx++) {

            // 2.) AGOR
            std::uint16_t quotient[]  = {0, 0, 0, 0};
            std::uint16_t remainder[] = {0, 0, 0, 0};
            std::uint16_t k[]         = {0, 0, 0, 0};
            //std::int16_t dpcm_curr[]  = {0, 0, 0, 0};

            for(std::size_t ch = 0; ch < 4; ch++) {
                for(std::size_t it = 0; it < (bpp_a + 2); it++) {
                    if((N << k[ch]) < A[ch]) {
                        k[ch] = k[ch] + 1;
                    } else {
                        k[ch] = k[ch];
                    }
                }
                std::uint32_t lastBit;

                std::uint16_t absVal = 0;
                do {   // decode quotient: unary coding
                    lastBit = reader.fetchBit_buffered();
                    if(lastBit == 1) {   //
                        quotient[ch]++;
                    }
                } while(lastBit == 1 && quotient[ch] < unaryMaxWidth);

                /* m_unaryMaxWidth * '1' -> indicates binary coding of positive value */
                if(quotient[ch] >= unaryMaxWidth) {
                    // for(std::uint32_t n = 0; n < k_seed; n++) {   //decode remainder
                    //     lastBit = reader.fetchBit_buffered();
                    //     absVal  = absVal | ((std::uint16_t)lastBit << n); /* LSB first */
                    //     // absVal  = (absVal << 1) | lastBit; /* MSB first*/
                    // };
                    // absVal = reader.getMSBfirst(reader.fetchBits_buffered(k_seed), k_seed);
                    absVal = getMSBfirst(reader.fetchBits_buffered(k_seed), k_seed);
                    // absVal = reader.fetchBits_buffered(k_seed);
                    // auto temp = absVal;
                } else {
                    // if(k[ch] == 0) {
                    //     absVal = quotient[ch];
                    // } else {
                    // for(std::uint32_t n = 0; n < k[ch]; n++) {   //decode remainder
                    //     lastBit       = reader.fetchBit_buffered();
                    //     remainder[ch] = remainder[ch] | ((std::uint16_t)lastBit << n); /* LSB first*/
                    //     // remainder[ch] = (remainder[ch] << 1) | lastBit; /* MSB first*/
                    // };
                    // remainder[ch] = (uint16_t)reader.fetchBits_buffered(k[ch]);
                    // remainder[ch] = (uint16_t)reader.getMSBfirst(reader.fetchBits_buffered(k[ch]), k[ch]);
                    remainder[ch] = (uint16_t)getMSBfirst(reader.fetchBits_buffered(k[ch]), k[ch]);
                    // auto temp = remainder[ch];
                    // auto temp_2 = reader.getMSBfirst(temp, k[ch]);
                    // absVal = quotient[ch] * (1 << k[ch]) + remainder[ch];
                    absVal = (quotient[ch] << k[ch]) + remainder[ch];
                    // }
                }

                // TODO: OPPORTUNITY: unsigned absVal to signed dpcm can also be done in parallel
                YCCC_dpcm_h[4 * idx + ch] = DecoderBase::fromAbs(absVal);
            }

            A[0] += YCCC_dpcm_h[4 * idx + 0] > 0 ? YCCC_dpcm_h[4 * idx + 0] : -YCCC_dpcm_h[4 * idx + 0];
            A[1] += YCCC_dpcm_h[4 * idx + 1] > 0 ? YCCC_dpcm_h[4 * idx + 1] : -YCCC_dpcm_h[4 * idx + 1];
            A[2] += YCCC_dpcm_h[4 * idx + 2] > 0 ? YCCC_dpcm_h[4 * idx + 2] : -YCCC_dpcm_h[4 * idx + 2];
            A[3] += YCCC_dpcm_h[4 * idx + 3] > 0 ? YCCC_dpcm_h[4 * idx + 3] : -YCCC_dpcm_h[4 * idx + 3];

            N += 1;
            if(N >= N_threshold) {
                N >>= 1;
                A[0] >>= 1;
                A[1] >>= 1;
                A[2] >>= 1;
                A[3] >>= 1;
            }
            A[0] = A[0] < A_MIN ? A_MIN : A[0];
            A[1] = A[1] < A_MIN ? A_MIN : A[1];
            A[2] = A[2] < A_MIN ? A_MIN : A[2];
            A[3] = A[3] < A_MIN ? A_MIN : A[3];
        }

#    endif

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point end_parsing = std::chrono::steady_clock::now();
#    endif

        // calculate all pixels in first column
        // CALCULATE ALL YCCC from DPCM
        for(std::size_t row = 1; row < height; row++) {
            // 4*width (because there are 4 channels of each width (which is half of the actual image width))
            auto idx_curr = row * (4 * width);
            auto idx_prev = (row - 1) * (4 * width);

            // YCCC_prev_debug[idx_curr + 0] = YCCC_h[idx_prev + 0];
            // YCCC_prev_debug[idx_curr + 1] = YCCC_h[idx_prev + 1];
            // YCCC_prev_debug[idx_curr + 2] = YCCC_h[idx_prev + 2];
            // YCCC_prev_debug[idx_curr + 3] = YCCC_h[idx_prev + 3];
            YCCC_h[idx_curr + 0] = YCCC_h[idx_prev + 0] + YCCC_dpcm_h[idx_curr + 0];
            YCCC_h[idx_curr + 1] = YCCC_h[idx_prev + 1] + YCCC_dpcm_h[idx_curr + 1];
            YCCC_h[idx_curr + 2] = YCCC_h[idx_prev + 2] + YCCC_dpcm_h[idx_curr + 2];
            YCCC_h[idx_curr + 3] = YCCC_h[idx_prev + 3] + YCCC_dpcm_h[idx_curr + 3];
        }

        //***************************************************
        // STEP 9: Write host data to device buffers
        //***************************************************

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_dpcm_2_bayer_memory = std::chrono::steady_clock::now();
#    endif

        // write decoded DPCM values to global GPU memory
        status = clEnqueueWriteBuffer(
           cmdQueue,
           YCCC_dpcm_d,
           CL_FALSE,
           0,
           datasize_YCCC_dpcm_d,
           YCCC_dpcm_h.data(),
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);
        // write full YCCC values (first row already calculated) to global GPU memory
        status = clEnqueueWriteBuffer(cmdQueue, YCCC_d, CL_FALSE, 0, datasize_YCCC_d, YCCC_h.data(), 0, NULL, NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_dpcm_2_bayer_memory = std::chrono::steady_clock::now();
#    endif

        //***************************************************
        // STEP 10: Set the kernel arguments for dpcm to full YCCC calculation
        //***************************************************

        // Set the Argument values
        status = clSetKernelArg(ckDpcmAcrossRows, 0, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 1, sizeof(cl_mem), (void*)&YCCC_dpcm_d);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 2, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        status |= clSetKernelArg(ckDpcmAcrossRows, 3, sizeof(cl_int), (void*)&height);
        evaluateReturnStatus(status);

        // std::vector<std::int16_t> YCCC_h_gt(4 * (height * width));
        // for(std::size_t i = 0; i < 4 * (height * width); i++) {
        //     YCCC_h_gt[i] = YCCC_h[i];
        // }

        // calculate all pixels in each row
        // printf("Pseudo GPU execution\n");
        // for(std::size_t row = 0; row < height; row++) {

        //     auto row_offset = row * (4 * width);
        //     // printf(
        //     //    "Nr of rows: %lld, Nr of columns: %lld, Global ID: %lld, Local ID: %lld, Group ID: %lld, Curr row: %lld, Row "
        //     //    "offset: "
        //     //    "%lld\n",
        //     //    height,
        //     //    width,
        //     //    row,
        //     //    row%128,
        //     //    row/128,
        //     //    row,
        //     //    row_offset);
        //     for(std::size_t col = 1; col < width; col++) {
        //         auto idx_curr                 = row_offset + 4 * col;
        //         auto idx_prev                 = row_offset + 4 * (col - 1);
        //         YCCC_prev_debug[idx_curr + 0] = YCCC_h_gt[idx_prev + 0];
        //         YCCC_prev_debug[idx_curr + 1] = YCCC_h_gt[idx_prev + 1];
        //         YCCC_prev_debug[idx_curr + 2] = YCCC_h_gt[idx_prev + 2];
        //         YCCC_prev_debug[idx_curr + 3] = YCCC_h_gt[idx_prev + 3];
        //         YCCC_h_gt[idx_curr + 0]       = YCCC_h_gt[idx_prev + 0] + YCCC_dpcm_h[idx_curr + 0];
        //         YCCC_h_gt[idx_curr + 1]       = YCCC_h_gt[idx_prev + 1] + YCCC_dpcm_h[idx_curr + 1];
        //         YCCC_h_gt[idx_curr + 2]       = YCCC_h_gt[idx_prev + 2] + YCCC_dpcm_h[idx_curr + 2];
        //         YCCC_h_gt[idx_curr + 3]       = YCCC_h_gt[idx_prev + 3] + YCCC_dpcm_h[idx_curr + 3];
        //     }
        // }
        // printf("Pseudo GPU execution END\n");

        //***************************************************
        // STEP 11: Enqueue the kernel for execution
        //***************************************************

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_dpcm_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        // Launch kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckDpcmAcrossRows,
           1,
           NULL,
           &szGlobalWorkSize,
           &szLocalWorkSize,
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_dpcm_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        //***************************************************
        // STEP 12: Read the output buffer back to the host
        //***************************************************
        // Synchronous/blocking read of results
        // Read back full YCCC values
        // TODO: remove this because it is not actually necesarry
        // status = clEnqueueReadBuffer(cmdQueue, YCCC_d, CL_TRUE, 0, datasize_YCCC_d, YCCC_h.data(), 0, NULL, NULL);
        // evaluateReturnStatus(status);

        //***************************************************
        // STEP 13: Set the kernel arguments for yccc to bayerGB conversion
        //***************************************************

        status = clSetKernelArg(ckYcccToBayerGB, 0, sizeof(cl_mem), (void*)&YCCC_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 1, sizeof(cl_mem), (void*)&bayerGB_d);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 2, sizeof(cl_int), (void*)&lossyBits);
        evaluateReturnStatus(status);
        status = clSetKernelArg(ckYcccToBayerGB, 3, sizeof(cl_int), (void*)&width);
        evaluateReturnStatus(status);
        cl_int nrOfPixels = height * width;
        status            = clSetKernelArg(ckYcccToBayerGB, 4, sizeof(cl_int), (void*)&nrOfPixels);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_yccc_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

        // Launch kernel
        status = clEnqueueNDRangeKernel(
           cmdQueue,
           ckYcccToBayerGB,
           1,
           NULL,
           &szGlobalWorkSize_yccc_to_bayer,
           &szLocalWorkSize_yccc_to_bayer,
           0,
           NULL,
           NULL);
        evaluateReturnStatus(status);

#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_yccc_2_bayer_kernel = std::chrono::steady_clock::now();
#    endif

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point begin_yccc_2_bayer_memory = std::chrono::steady_clock::now();
#    endif
        // Read back BayerGB values
        status = clEnqueueReadBuffer(cmdQueue, bayerGB_d, CL_TRUE, 0, datasize_BayerGB, bayerGB, 0, NULL, NULL);
        evaluateReturnStatus(status);
        // Block until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed
#    ifdef TIMING_EN
        clFinish(cmdQueue);
        std::chrono::steady_clock::time_point end_yccc_2_bayer_memory = std::chrono::steady_clock::now();
#    endif

        printf("GPU execution finished\n");

        // char outputFile[200];
        // char txt[200];
        // sprintf(outputFile, "C:/DATA/Repos/OMLS_Masters_SW/images/lite_dataset/dump/gpu_DPCM.txt");
        // std::ofstream wf(outputFile, std::ios::out);
        // if(!wf) {
        //     char msg[200];
        //     sprintf(msg, "Cannot open specified file: %s", outputFile);
        //     throw std::runtime_error(msg);
        // }
        // std::sprintf(txt, "%zu %zu pred YCCC, curr YCCC, dpcm\n", width, height);
        // wf << txt;
        // int16_t mask_Y      = (1 << (bpp_a + 2)) - 1;   // 0x03FF
        // int16_t mask_C      = (1 << (bpp_a + 1)) - 1;   // 0x01FF;
        // int16_t mask_Y_dpcm = (1 << (bpp_a + 3)) - 1;   // 0x07FF;
        // int16_t mask_C_dpcm = (1 << (bpp_a + 2)) - 1;   // 0x03FF;

        // for(std::size_t i = 0; i < width * height; i++) {
        //     std::sprintf(
        //        txt,
        //        "%4zu %4zu : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX : %.4hX %.4hX %.4hX %.4hX\n",
        //        i / width,
        //        i % width,
        //        (YCCC_prev_debug[4 * i + 0] & mask_Y),
        //        (YCCC_prev_debug[4 * i + 1] & mask_C),
        //        (YCCC_prev_debug[4 * i + 2] & mask_C),
        //        (YCCC_prev_debug[4 * i + 3] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 0] & mask_Y),
        //        //    (YCCC_h_gt[4 * i + 1] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 2] & mask_C),
        //        //    (YCCC_h_gt[4 * i + 3] & mask_C),
        //        (YCCC_h[4 * i + 0] & mask_Y),
        //        (YCCC_h[4 * i + 1] & mask_C),
        //        (YCCC_h[4 * i + 2] & mask_C),
        //        (YCCC_h[4 * i + 3] & mask_C),
        //        (YCCC_dpcm_h[4 * i + 0] & mask_Y_dpcm),
        //        (YCCC_dpcm_h[4 * i + 1] & mask_C_dpcm),
        //        (YCCC_dpcm_h[4 * i + 2] & mask_C_dpcm),
        //        (YCCC_dpcm_h[4 * i + 3] & mask_C_dpcm));
        //     wf << txt;
        // }
        // wf.close();

        //***************************************************
        // STEP 13: Release OpenCL resources
        //***************************************************

        if(ckDpcmAcrossRows)
            clReleaseKernel(ckDpcmAcrossRows);
        if(cpProgram)
            clReleaseProgram(cpProgram);
        if(cmdQueue)
            clReleaseCommandQueue(cmdQueue);
        if(context)
            clReleaseContext(context);
        if(YCCC_d)
            clReleaseMemObject(YCCC_d);
        if(YCCC_dpcm_d)
            clReleaseMemObject(YCCC_dpcm_d);

        if(ckYcccToBayerGB)
            clReleaseKernel(ckYcccToBayerGB);
        if(bayerGB_d)
            clReleaseMemObject(bayerGB_d);

        delete[] platforms;
        delete[] devices;

#    ifdef TIMING_EN
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl;
        std::cout << "OpenCL: Parallel decoding time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        std::cout << "OpenCL: Parsing time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parsing - begin_parsing).count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Memory transfer DPCM to YCCC time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_dpcm_2_bayer_memory - begin_dpcm_2_bayer_memory)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Kernel DPCM to YCCC time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_dpcm_2_bayer_kernel - begin_dpcm_2_bayer_kernel)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Kernel YCCC to BayerGB time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_yccc_2_bayer_kernel - begin_yccc_2_bayer_kernel)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << "OpenCL: Memory transfer YCCC to BayerGB time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_yccc_2_bayer_memory - begin_yccc_2_bayer_memory)
                        .count()
                  << "[ms]" << std::endl;
        std::cout << std::endl;
#    endif

        printf("OpenCL: Parallel decoding finished\n");
        return BASE_SUCCESS;
    }

#endif

    // template<typename T>
    static STATUS_t exportImage(
       const char* fileName,
       //    std::uint16_t* data,
       std::uint8_t* data,
       std::size_t data_size_bytes,
       std::uint64_t header,
       std::uint64_t roi,
       std::uint64_t timestamp)
    {
        std::ofstream wf(fileName, std::ios::out | std::ios::binary);
        if(!wf) {
            return BASE_CANNOT_OPEN_OUTPUT_FILE;
        }

        if(timestamp) {
            wf.write((char*)&timestamp, 8); /* Little endian order */
        }

        if(roi) {
            wf.write((char*)&roi, 8); /* Little endian order */
        }

        if(header) {
            wf.write((char*)&header, 8); /* Little endian order */
        }

        /* Little endian order*/
        wf.write((char*)data, data_size_bytes);

        wf.close();

        return BASE_SUCCESS;
    }

    // template<typename T>
    static STATUS_t exportImage(
       const char* folderName,
       const char* fileName,
       std::uint8_t* data,
       std::size_t data_size_bytes,
       std::uint64_t header,
       std::uint64_t roi,
       std::uint64_t timestamp)
    {
        createMissingDirectory(folderName);
        char path[500];
        sprintf_s(path, "%s/%s", folderName, fileName);

        return DecoderBase::exportImage(path, data, data_size_bytes, header, roi, timestamp);
    }

    static STATUS_t importBitstream(const char* fileName, std::unique_ptr<std::vector<std::uint8_t>>& outData);

    /**
   * Get DPCM value from Absolute value.
   */
    static std::int16_t fromAbs(std::uint16_t absValue);

    /**
     * Transform from YCdCmCo to Bayer GB color space
     *
     * Y  | Cd =>  Gb | Bl
     * ---+--- =>  ---+---
     * Cm | Co =>  Rd | Gr
     *
     */
    template<typename T>
    static STATUS_t YCCC_to_BayerGB(
       const std::int16_t y,
       const std::int16_t cd,
       const std::int16_t cm,
       const std::int16_t co,
       T& gb,
       T& b,
       T& r,
       T& gr,
       std::size_t lossyBits)
    {
        // clang-format off
    //std::int16_t gr_temp = (2 * y +  2 * cd +  4 * cm +  2 * co);
    //std::int16_t r_temp =  (2 * y +  2 * cd + -4 * cm +  2 * co);  
    //std::int16_t b_temp =  (2 * y +  2 * cd + -4 * cm + -6 * co);  
    //std::int16_t gb_temp = (2 * y + -6 * cd +  4 * cm +  2 * co); 
    //
    //std::uint16_t mask = 7 >> lossyBits;
    //if((gr_temp & mask) != 0 ){
    //    return BASE_DIVISION_ERROR;
    //}
    //if((b_temp & mask) != 0 ){
    //    return BASE_DIVISION_ERROR;
    //}
    //if((r_temp & mask) != 0 ){
    //    return BASE_DIVISION_ERROR;
    //}
    //if((gb_temp & mask) != 0 ){
    //    return BASE_DIVISION_ERROR;
    //}

    gr = (T)((std::int16_t)(2 * y +  2 * cd +  4 * cm +  2 * co) >> 3) << lossyBits;   // divide by 8
    r  = (T)((std::int16_t)(2 * y +  2 * cd + -4 * cm +  2 * co) >> 3) << lossyBits;   // divide by 8
    b  = (T)((std::int16_t)(2 * y +  2 * cd + -4 * cm + -6 * co) >> 3) << lossyBits;   // divide by 8
    gb = (T)((std::int16_t)(2 * y + -6 * cd +  4 * cm +  2 * co) >> 3) << lossyBits;   // divide by 8
        // clang-format on
        return BASE_SUCCESS;
    }

    static STATUS_t handleReturnValue(STATUS_t status);
};
