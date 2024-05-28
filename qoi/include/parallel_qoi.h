#ifndef PARALLEL_QOI
#define PARALLEL_QOI


#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include "kernel_loader.h"
#include "compact_types.h"

void *parallel_qoi_encode(const void *data, const qoi_desc *desc, int *out_len);
void parallel_process(unsigned char *bytes, int bytes_len, const unsigned char *pixels, int pixels_len, unsigned int *segment_lengths, const qoi_desc *desc, ocl_res_t *ocl);
static inline unsigned char *merge_segments(unsigned char *bytes, unsigned int *segment_lengths, const qoi_desc *desc, int *total_size);
int parallel_qoi_write(const char *filename, const void *data, const qoi_desc *desc);

// encode target image using opencl parallel computing
// returns the size of the data written or 0 on failure
 void *parallel_qoi_encode(const void *data, const qoi_desc *desc, int *out_len){
    // prepare opencl
    ocl_res_t ocl;
    init_opencl(&ocl);

    const char *kernel_source = "kernels/codec.cl";
    const char *options = "-D SET_ME=1234";
    const char *kernel_name = "encode";

    load_kernel_code(&ocl, kernel_source);
    create_program(&ocl);
    build_program(&ocl, options);
    create_kernel(&ocl, kernel_name);


	// prepare image and kernels
    int max_size =
        desc->width * desc->height * (desc->channels + 1) +
        QOI_HEADER_SIZE + sizeof(qoi_padding);

    // calloc instead of QOI_MALLOC for cleanly separated data after compression
    unsigned char *bytes = (unsigned char *) calloc(max_size, sizeof(unsigned char));

    const unsigned char *pixels = (const unsigned char *)data;
    int px_len = desc->width * desc->height * desc->channels;

    // length of each compressed segment
	unsigned int segment_lengths[desc->height];

    // add header info
    // merge compressed segments
    // remove artifacts and redundant chunks
    parallel_process(bytes, max_size, pixels, px_len, segment_lengths, desc, &ocl);

    int merged_size;
    clock_t begin = clock();
    unsigned char *merged = merge_segments(bytes, segment_lengths, desc, &merged_size);
    free(bytes);
    clock_t end = clock();


    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("OpenCL QOI encoder merge time: %lfs\n", time_spent);

    // don't need segments anymore
    *out_len = merged_size;
        
    return merged;
}

void parallel_process(unsigned char *bytes, int bytes_len, const unsigned char *pixels, int pixels_len, unsigned int *segment_lengths,
    const qoi_desc *desc, ocl_res_t *ocl) {
    
    // bind opencl buffers and launch kernel
    cl_mem pixel_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, pixels_len * sizeof(unsigned char), NULL, NULL);
    cl_mem bytes_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, bytes_len * sizeof(unsigned char), NULL, NULL);
    cl_mem segment_lengths_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, desc->height * sizeof(unsigned int), NULL, NULL);

    // TODO: adjust work item sizes in case img_height > CL_DEVICE_MAX_WORK_ITEM_SIZES
    clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void*)&pixel_buffer);
    clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void*)&bytes_buffer);
    clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), (void*)&segment_lengths_buffer);
    clSetKernelArg(ocl->kernel, 3, sizeof(int), (void*)&desc->width);
    clSetKernelArg(ocl->kernel, 4, sizeof(int), (void*)&desc->channels);

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #pragma GCC diagnostic ignored "-Wint-conversion"
    cl_command_queue command_queue = clCreateCommandQueue(ocl->context, ocl->device_id, CL_QUEUE_PROFILING_ENABLE, NULL);
    #pragma GCC diagnostic pop

    // pixels --> pixel_buffer
    clEnqueueWriteBuffer(
        command_queue,
        pixel_buffer,
        CL_FALSE,
        0,
        pixels_len * sizeof(unsigned char),
        pixels,
        0,
        NULL,
        NULL
    );
    
    // apply kernel to every line (segment) of the image
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        ocl->kernel,
        1,
        NULL,
        &desc->height,
        NULL,
        0,
        NULL,
        &event
    );

    // bytes_buffer --> bytes
    clEnqueueReadBuffer(
        command_queue,
        bytes_buffer,
        CL_TRUE,
        0,
        bytes_len * sizeof(unsigned char),
        bytes,
        0,
        NULL,
        NULL
    );

    // segments_buffer --> segments
    clEnqueueReadBuffer(
        command_queue,
        segment_lengths_buffer,
        CL_TRUE,
        0,
        desc->height * sizeof(unsigned int),
        segment_lengths,
        0,
        NULL,
        NULL
    );

    // measure kernel execution time
    clWaitForEvents(1, &event);
    clFinish(command_queue);

    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double ns = time_end-time_start;
    printf("OpenCL kernel execution time: %lfs\n", ns/1.0e9);

    // release resources
    clReleaseKernel(ocl->kernel);
    clReleaseProgram(ocl->program);
    clReleaseContext(ocl->context);
    clReleaseDevice(ocl->device_id);

    clReleaseMemObject(pixel_buffer);
    clReleaseMemObject(bytes_buffer);
    clReleaseMemObject(segment_lengths_buffer);
}

static inline unsigned char *merge_segments(unsigned char *bytes, unsigned int *segment_lengths, const qoi_desc *desc, int *total_size){
    int merged_size = 0;
    for (int i = 0; i < desc->height; i++){
        merged_size += segment_lengths[i];
    }
    merged_size += QOI_HEADER_SIZE + sizeof(qoi_padding);
    unsigned char *merged = (unsigned char*)calloc(merged_size, sizeof(unsigned char));

    int p = 0;

    // add header
    qoi_write_32(merged, &p, QOI_MAGIC);
    qoi_write_32(merged, &p, desc->width);
    qoi_write_32(merged, &p, desc->height);
    merged[p++] = desc->channels;
    merged[p++] = desc->colorspace;

    // merge segments
    int k;
    for (int i = 0; i < desc->height; i++){
        k = i * desc->width * (desc->channels + 1);
        memcpy(&merged[p], &bytes[k], segment_lengths[i]);
        p += segment_lengths[i];
    }

    *total_size = merged_size;
    return merged;
}

int parallel_qoi_write(const char *filename, const void *data, const qoi_desc *desc){
    FILE *f = fopen(filename, "wb");
    int size, err;
    void *encoded;

    if (!f) {
        return 0;
    }

    encoded = parallel_qoi_encode(data, desc, &size);

    if (!encoded) {
        fclose(f);
        return 0;
    }

    fwrite(encoded, 1, size, f);
    fflush(f);
    err = ferror(f);
    fclose(f);

    QOI_FREE(encoded);
    return err ? 0 : size;
}

#endif