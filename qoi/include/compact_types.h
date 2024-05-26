#include <CL/cl.h>

#ifndef COMPACT_TYPES
#define COMPACT_TYPES

typedef struct ocl_res{
    cl_int err;
    cl_uint n_platforms;
    cl_uint n_devices;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_program program;
    const char *kernel_code;
    cl_kernel kernel;
} ocl_res_t;

void get_platform(ocl_res_t *ocl);
void get_device(ocl_res_t *ocl);
void create_context(ocl_res_t *ocl);
void load_kernel_code(ocl_res_t *ocl, const char *path);
void create_program(ocl_res_t *ocl);
void build_program(ocl_res_t *ocl, const char *opitons);
void create_kernel(ocl_res_t *ocl, const char *kernel_name);
void init_opencl(ocl_res_t *ocl);

const char *get_error_msg(int error);

#endif