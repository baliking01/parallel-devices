#include "compact_types.h"
#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

void get_platform(ocl_res_t *ocl){
    ocl->err = clGetPlatformIDs(1, &ocl->platform_id, &ocl->n_platforms);
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d :: %s\n", ocl->err, get_error_msg(ocl->err));
        exit(1);
    }
}

void get_device(ocl_res_t *ocl){
    ocl->err = clGetDeviceIDs(
        ocl->platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &ocl->device_id,
        &ocl->n_devices
    );
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d :: %s\n", ocl->err, get_error_msg(ocl->err));
        exit(1);
    }
}

void create_context(ocl_res_t *ocl){
	ocl->context = clCreateContext(NULL, ocl->n_devices, &ocl->device_id, NULL, NULL, &ocl->err);
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Error creating context. Error code: %d :: %s\n", ocl->err, get_error_msg(ocl->err));
        exit(1);
    }
}

void load_kernel_code(ocl_res_t *ocl, const char* path){
	int error_code;
	ocl->kernel_code = load_kernel_source(path, &error_code);
    if (error_code != 0) {
        printf("[ERROR] Source code loading error!\n");
        exit(1);
    }
}

void create_program(ocl_res_t *ocl){
	ocl->program = clCreateProgramWithSource(ocl->context, 1, &ocl->kernel_code, NULL, &ocl->err);
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Error creating program. Error code: %d\n :: %s", ocl->err, get_error_msg(ocl->err));
        exit(1);
    }
}

void build_program(ocl_res_t *ocl, const char *options){
    ocl->err = clBuildProgram(
        ocl->program,
        1,
        &ocl->device_id,
        options,
        NULL,
        NULL
    );
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Build error! Code: %d :: %s\n", ocl->err, get_error_msg(ocl->err));
        size_t real_size;

        ocl->err = clGetProgramBuildInfo(
            ocl->program,
            ocl->device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );

        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        ocl->err = clGetProgramBuildInfo(
            ocl->program,
            ocl->device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );

        build_log[real_size] = 0;  // Ensure null termination
        printf("Real size : %zu\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        exit(1);
    }

    size_t sizes_param[10];
    size_t real_size;
    ocl->err = clGetProgramInfo(
        ocl->program,
        CL_PROGRAM_BINARY_SIZES,
        sizeof(sizes_param),
        sizes_param,
        &real_size
    );
    //printf("Program info: \n");
    //printf("Real size   : %zu\n", real_size);
    //printf("Binary size : %zu\n", sizes_param[0]);
}


void create_kernel(ocl_res_t *ocl, const char *kernel_name){
	ocl->kernel = clCreateKernel(ocl->program, kernel_name, &ocl->err);
    if (ocl->err != CL_SUCCESS) {
        printf("[ERROR] Error creating kernel. Error code: %d :: %s\n", ocl->err, get_error_msg(ocl->err));
        exit(1);
    }
}

void init_opencl(ocl_res_t *ocl){
	get_platform(ocl);
	get_device(ocl);
	create_context(ocl);
}

const char *get_error_msg(cl_int error) {
	switch(error){
	    // run-time and JIT compiler errors
	    case 0: return "CL_SUCCESS";
	    case -1: return "CL_DEVICE_NOT_FOUND";
	    case -2: return "CL_DEVICE_NOT_AVAILABLE";
	    case -3: return "CL_COMPILER_NOT_AVAILABLE";
	    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	    case -5: return "CL_OUT_OF_RESOURCES";
	    case -6: return "CL_OUT_OF_HOST_MEMORY";
	    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	    case -8: return "CL_MEM_COPY_OVERLAP";
	    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	    case -11: return "CL_BUILD_PROGRAM_FAILURE";
	    case -12: return "CL_MAP_FAILURE";
	    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	    case -16: return "CL_LINKER_NOT_AVAILABLE";
	    case -17: return "CL_LINK_PROGRAM_FAILURE";
	    case -18: return "CL_DEVICE_PARTITION_FAILED";
	    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	    // compile-time errors
	    case -30: return "CL_INVALID_VALUE";
	    case -31: return "CL_INVALID_DEVICE_TYPE";
	    case -32: return "CL_INVALID_PLATFORM";
	    case -33: return "CL_INVALID_DEVICE";
	    case -34: return "CL_INVALID_CONTEXT";
	    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	    case -36: return "CL_INVALID_COMMAND_QUEUE";
	    case -37: return "CL_INVALID_HOST_PTR";
	    case -38: return "CL_INVALID_MEM_OBJECT";
	    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	    case -40: return "CL_INVALID_IMAGE_SIZE";
	    case -41: return "CL_INVALID_SAMPLER";
	    case -42: return "CL_INVALID_BINARY";
	    case -43: return "CL_INVALID_BUILD_OPTIONS";
	    case -44: return "CL_INVALID_PROGRAM";
	    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	    case -46: return "CL_INVALID_KERNEL_NAME";
	    case -47: return "CL_INVALID_KERNEL_DEFINITION";
	    case -48: return "CL_INVALID_KERNEL";
	    case -49: return "CL_INVALID_ARG_INDEX";
	    case -50: return "CL_INVALID_ARG_VALUE";
	    case -51: return "CL_INVALID_ARG_SIZE";
	    case -52: return "CL_INVALID_KERNEL_ARGS";
	    case -53: return "CL_INVALID_WORK_DIMENSION";
	    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	    case -56: return "CL_INVALID_GLOBAL_OFFSET";
	    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	    case -58: return "CL_INVALID_EVENT";
	    case -59: return "CL_INVALID_OPERATION";
	    case -60: return "CL_INVALID_GL_OBJECT";
	    case -61: return "CL_INVALID_BUFFER_SIZE";
	    case -62: return "CL_INVALID_MIP_LEVEL";
	    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	    case -64: return "CL_INVALID_PROPERTY";
	    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	    case -66: return "CL_INVALID_COMPILER_OPTIONS";
	    case -67: return "CL_INVALID_LINKER_OPTIONS";
	    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

	    // extension errors
	    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	    default: return "Unknown OpenCL error";
    }
}