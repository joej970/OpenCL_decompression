
// #include <cstdint>

__kernel void dpcm_across_rows(__global short* YCCC, __global short* YCCC_dpcm, int nrOfColumns, int nrOfRows)
{
    int global_id = get_global_id(0);
    int local_id  = get_local_id(0);
    int group_id  = get_group_id(0);

    int curr_row = global_id;
    if(curr_row < nrOfRows) {
        int row_offset = curr_row * (4 * nrOfColumns);
        // printf(
        //    "Nr of rows: %d, Nr of columns: %d, Global ID: %d, Local ID: %d, Group ID: %d, Curr row: %d, Row offset: "
        //    "%d\n",
        //    nrOfRows,
        //    nrOfColumns,
        //    global_id,
        //    local_id,
        //    group_id,
        //    curr_row,
        //    row_offset);
        // printf(
        //    "Global ID: %d, Start idx: %d, End idx: %d\n",
        //    global_id,
        //    row_offset + 4 * 1,
        //    row_offset + 4 * (nrOfColumns - 1));
        for(int col = 1; col < nrOfColumns; col++) {
            int idx_curr = row_offset + 4 * col;
            int idx_prev = row_offset + 4 * (col - 1);

            YCCC[idx_curr + 0] = YCCC[idx_prev + 0] + YCCC_dpcm[idx_curr + 0];
            YCCC[idx_curr + 1] = YCCC[idx_prev + 1] + YCCC_dpcm[idx_curr + 1];
            YCCC[idx_curr + 2] = YCCC[idx_prev + 2] + YCCC_dpcm[idx_curr + 2];
            YCCC[idx_curr + 3] = YCCC[idx_prev + 3] + YCCC_dpcm[idx_curr + 3];
        }
    }
}

inline void kh_YCCC_to_BayerGB_8bit(
   __global short* y,
   __global short* cd,
   __global short* cm,
   __global short* co,
   __global uchar* gb,
   __global uchar* b,
   __global uchar* r,
   __global uchar* gr,
   uint* lossyBits)
{

    // clang-format off
    *gr = (uchar)((short)(2 * (*y) +  2 * (*cd) +  4 * (*cm) +  2 * (*co)) >> 3) << *lossyBits;   // divide by 8
    *r  = (uchar)((short)(2 * (*y) +  2 * (*cd) + -4 * (*cm) +  2 * (*co)) >> 3) << *lossyBits;   // divide by 8
    *b  = (uchar)((short)(2 * (*y) +  2 * (*cd) + -4 * (*cm) + -6 * (*co)) >> 3) << *lossyBits;   // divide by 8
    *gb = (uchar)((short)(2 * (*y) + -6 * (*cd) +  4 * (*cm) +  2 * (*co)) >> 3) << *lossyBits;   // divide by 8
       // clang-format on
}

inline short kh_fromAbs(ushort absValue)
{
    if(absValue % 2 == 0) {
        return absValue / 2;
    } else {
        return -((absValue + 1) / 2);
    }
};

__kernel void
   yccc_to_bayergb_8bit(__global short* YCCC, __global uchar* bayerGB, uint lossyBits, uint width, uint nrOfPixels)
{
    int global_id = get_global_id(0);

    int curr_idx = global_id;
    if(curr_idx < nrOfPixels) {
        int idxGB = (curr_idx / width) * width * 4 + 2 * (curr_idx % width);
        kh_YCCC_to_BayerGB_8bit(
           &YCCC[4 * curr_idx + 0],   //
           &YCCC[4 * curr_idx + 1],
           &YCCC[4 * curr_idx + 2],
           &YCCC[4 * curr_idx + 3],
           &bayerGB[idxGB],
           &bayerGB[idxGB + 1],
           &bayerGB[idxGB + 2 * width],
           &bayerGB[idxGB + 2 * width + 1],
           &lossyBits);
    }
}

inline ushort kh_fetchBit(
   __global uchar* bitStream,
   //    ulong bitStream_size,
   ulong* bitsReadFromByte,
   ulong* byteIdx,
   uchar* byte)
{
    // if(*byteIdx == bitStream_size) {
    //     printf("All bytes have been read.\n");
    // }
    if(*bitsReadFromByte == 8) {
        *bitsReadFromByte = 0;
        // (byteIdx)++;
        *byteIdx = *byteIdx + 1;
        *byte    = bitStream[*byteIdx];   //update only if it all bits have been read
    }
    // todo: consider changing bit order so that there is only bitshift for 1 bit >> 1
    ushort bit = (*byte >> (7 - *bitsReadFromByte)) & (ushort)1;
    // bitsReadFromByte++;
    *bitsReadFromByte = *bitsReadFromByte + 1;
    return bit;
}

__kernel void bitstream_to_dpcm(
   __global uchar* bitstream,
   __global uint* pixelsInBlock,
   __global short* YCCC_dpcm,   // can be local
   __global short* YCCC,
   ushort unaryMaxWidth,
   //    ulong totalPixels,
   ulong bpp,
   ulong groupByteOffset,
   int nrOfColumns,
   int nrOfRows,
   int nrOfRowsInBlock)
{

    int global_id = get_global_id(0);
    int local_id  = get_local_id(0);

    int node_offset   = nrOfRowsInBlock * nrOfColumns * global_id;

    ulong currentByteOffset = groupByteOffset * global_id;


    // printf(
    //    "kernel: bitstream_to_dpcm: GID: %d, LID: %d |, number of rows in a block: %d, group byte off: "
    //    "%lu, "
    //    "curr byte off: %lu, node off: %d, pixInBlock: %u, quadsInBlock: %u \n",
    //    global_id,
    //    local_id,
    //    nrOfRowsInBlock,
    //    groupByteOffset,
    //    currentByteOffset,
    //    node_offset,
    //    pixelsInBlock[global_id],
    //    pixelsInBlock[global_id]/4);

    /* Variables for bitstream reading */
    // ulong bitStreamSize,
    ulong bitsReadFromByte = 0;
    ulong byteIdx          = 0;
    uchar byte             = bitstream[currentByteOffset];   // load first byte

    /* Other variable */

    uint N_threshold = 8;
    uint A_init      = 32;

    // uint A[]    = {A_init, A_init, A_init, A_init};
    uint4 A    = {A_init, A_init, A_init, A_init};
    uint N      = 4 + 1 - 1;
    uint k_seed = bpp + 3;   // max 12 BPP + 3 = 15

    ushort4 quotient  = {unaryMaxWidth, unaryMaxWidth, unaryMaxWidth, unaryMaxWidth};
    ushort4 remainder = {0, 0, 0, 0};
    ushort4 k         = {0, 0, 0, 0};
    ushort4 absVal    = {0, 0, 0, 0};
    // ushort quotient[]  = {unaryMaxWidth, unaryMaxWidth, unaryMaxWidth, unaryMaxWidth};
    // ushort remainder[] = {0, 0, 0, 0};
    // ushort k[]         = {0, 0, 0, 0};
    // ushort absVal[]    = {0, 0, 0, 0};

    // Get DPCM
    for(ulong idx = 0; idx < pixelsInBlock[global_id]/4; idx++) {
        // for(ulong idx = 0; idx < totalPixels; idx++) {
        // for(std::size_t idx = 1; idx < height * width; idx++) {

        for(uchar ch = 0; ch < 4; ch++) {
            for(uchar it = 0; it < (bpp + 2); it++) {
                // if((N << k[ch]) < A[ch]) {
                if((N << k[ch]) < A[ch]) {
                    k[ch] = k[ch] + 1;
                } else {
                    k[ch] = k[ch];
                }
            }
            uint lastBit;

            do {   // decode quotient: unary coding
                lastBit = kh_fetchBit(bitstream, &bitsReadFromByte, &byteIdx, &byte);
                if(lastBit == 1) {   //
                    quotient[ch]++;
                }
            } while(lastBit == 1 && quotient[ch] < unaryMaxWidth);

            /* m_unaryMaxWidth * '1' -> indicates binary coding of positive value */
            if(quotient[ch] >= unaryMaxWidth) {
                for(uint n = 0; n < k_seed; n++) {   //decode remainder
                    lastBit    = kh_fetchBit(bitstream, &bitsReadFromByte, &byteIdx, &byte);
                    absVal[ch] = absVal[ch] | ((ushort)lastBit << n); /* LSB first */
                    // absVal  = (absVal << 1) | lastBit; /* MSB first*/
                };
            } else {
                for(uint n = 0; n < k[ch]; n++) {   //decode remainder
                    lastBit       = kh_fetchBit(bitstream, &bitsReadFromByte, &byteIdx, &byte);
                    remainder[ch] = remainder[ch] | ((ushort)lastBit << n); /* LSB first*/
                    // remainder[ch] = (remainder[ch] << 1) | lastBit; /* MSB first*/
                };
                absVal[ch] = quotient[ch] * (1 << k[ch]) + remainder[ch];
            }

        }

        // short YCCC_dpcm_local[4];
        short4 YCCC_dpcm_local;
        YCCC_dpcm_local[0] = kh_fromAbs(absVal[0]);
        YCCC_dpcm_local[1] = kh_fromAbs(absVal[1]);
        YCCC_dpcm_local[2] = kh_fromAbs(absVal[2]);
        YCCC_dpcm_local[3] = kh_fromAbs(absVal[3]);

        // printf(
        //    "kernel [%d | %d]: idx: %lu/%d | byteIdx: %lu/%lu | dpcm[0]: %d, dpcm[1]: %d, dpcm[2]: %d, dpcm[3]: %d\n",
        //    global_id,
        //    local_id,
        //    idx,
        //    pixelsInBlock,
        //    byteIdx,
        //    groupByteOffset * (global_id + 1),
        //    YCCC_dpcm_local[0],
        //    YCCC_dpcm_local[1],
        //    YCCC_dpcm_local[2],
        //    YCCC_dpcm_local[3]);

        // Write to global memory
        // YCCC_dpcm[node_offset + 4 * idx + 0] = YCCC_dpcm_local[0];
        // YCCC_dpcm[node_offset + 4 * idx + 1] = YCCC_dpcm_local[1];
        // YCCC_dpcm[node_offset + 4 * idx + 2] = YCCC_dpcm_local[2];
        // YCCC_dpcm[node_offset + 4 * idx + 3] = YCCC_dpcm_local[3];

        vstore4(YCCC_dpcm_local, node_offset/4 + idx, YCCC_dpcm);


        A[0] += YCCC_dpcm_local[0] > 0 ? YCCC_dpcm_local[0] : -YCCC_dpcm_local[0];
        A[1] += YCCC_dpcm_local[1] > 0 ? YCCC_dpcm_local[1] : -YCCC_dpcm_local[1];
        A[2] += YCCC_dpcm_local[2] > 0 ? YCCC_dpcm_local[2] : -YCCC_dpcm_local[2];
        A[3] += YCCC_dpcm_local[3] > 0 ? YCCC_dpcm_local[3] : -YCCC_dpcm_local[3];
        
        N += 1;
        if(N >= N_threshold) {
            N >>= 1;
            A >>= 1;
            // A[0] >>= 1;
            // A[1] >>= 1;
            // A[2] >>= 1;
            // A[3] >>= 1;
        }

        quotient = 0;
        remainder = 0;
        k = 0;
        absVal = 0;


        // quotient[0]  = 0;
        // quotient[1]  = 0;
        // quotient[2]  = 0;
        // quotient[3]  = 0;
        // remainder[0] = 0;
        // remainder[1] = 0;
        // remainder[2] = 0;
        // remainder[3] = 0;
        // k[0]         = 0;
        // k[1]         = 0;
        // k[2]         = 0;
        // k[3]         = 0;
        // absVal[0]    = 0;
        // absVal[1]    = 0;
        // absVal[2]    = 0;
        // absVal[3]    = 0;
    }



    YCCC[node_offset + 0] = YCCC_dpcm[node_offset + 0];
    YCCC[node_offset + 1] = YCCC_dpcm[node_offset + 1];
    YCCC[node_offset + 2] = YCCC_dpcm[node_offset + 2];
    YCCC[node_offset + 3] = YCCC_dpcm[node_offset + 3];
}

__kernel void first_column_all_rows(
   __global short* YCCC_dpcm,
   __global short* YCCC,
   __global uint* pixelsInBlock,
   int nrOfColumns,
   int nrOfRows,
   int nrOfRowsInBlock)
{
    int global_id = get_global_id(0);
    int local_id  = get_local_id(0);
    // int group_id  = get_group_id(0);

    // calculate all pixels in first column
    // CALCULATE ALL YCCC from DPCM

    // int node_offset = nrOfRows/get_local_size(0) * local_id;
    int node_offset = nrOfRows / nrOfRowsInBlock * nrOfColumns * global_id;

    int last_thread_id = nrOfRows / nrOfRowsInBlock;

    uint rowsToEvaluate = (pixelsInBlock[global_id]/4)/nrOfColumns;

    // printf(
    //    "kernel: first_col_all_rows | GID: %d, LID: %d, Node off: %d, rows in block: %d, Nr of rows: %d, Nr of pix in block: %u, nr of rows to evaluate %d\n",
    //    global_id,
    //    local_id,
    //    node_offset,
    //    nrOfRowsInBlock,
    //    nrOfRows,
    //    pixelsInBlock[global_id],
    //    rowsToEvaluate);

    // if(global_id == last_thread_id) {
    //     int rowsToCheck = nrOfRows % nrOfRowsInBlock;
    //     printf("Last thread id: %d, rows to check: %d\n", last_thread_id, rowsToCheck);
    //     for(int row = 1; row < rowsToCheck; row++) {
    //         // 4*width (because there are 4 channels of each width (which is half of the actual image width))
    //         int idx_curr = node_offset + row * (4 * nrOfColumns);
    //         int idx_prev = node_offset + (row - 1) * (4 * nrOfColumns);

    //         YCCC[idx_curr + 0] = YCCC[idx_prev + 0] + YCCC_dpcm[idx_curr + 0];
    //         YCCC[idx_curr + 1] = YCCC[idx_prev + 1] + YCCC_dpcm[idx_curr + 1];
    //         YCCC[idx_curr + 2] = YCCC[idx_prev + 2] + YCCC_dpcm[idx_curr + 2];
    //         YCCC[idx_curr + 3] = YCCC[idx_prev + 3] + YCCC_dpcm[idx_curr + 3];
    //     }
    // } else {

    for(uint row = 1; row < rowsToEvaluate; row++) {
        // 4*width (because there are 4 channels of each width (which is half of the actual image width))
        int idx_curr = node_offset + row * (4 * nrOfColumns);
        int idx_prev = node_offset + (row - 1) * (4 * nrOfColumns);

        YCCC[idx_curr + 0] = YCCC[idx_prev + 0] + YCCC_dpcm[idx_curr + 0];
        YCCC[idx_curr + 1] = YCCC[idx_prev + 1] + YCCC_dpcm[idx_curr + 1];
        YCCC[idx_curr + 2] = YCCC[idx_prev + 2] + YCCC_dpcm[idx_curr + 2];
        YCCC[idx_curr + 3] = YCCC[idx_prev + 3] + YCCC_dpcm[idx_curr + 3];
    }
    // }
}