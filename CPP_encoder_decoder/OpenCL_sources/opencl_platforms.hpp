
#pragma once

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

const char* getErrorString(cl_int error);
cl_int evaluateReturnStatus(cl_int result);
int identify_platforms();