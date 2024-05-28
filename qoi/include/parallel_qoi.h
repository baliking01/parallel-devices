#ifndef PARALLEL_QOI
#define PARALLEL_QOI

void *parallel_qoi_encode(const void *data, const qoi_desc *desc, int *out_len);
int parallel_qoi_write(const char *filename, const void *data, const qoi_desc *desc);

#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include "kernel_loader.h"
#include "compact_types.h"


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
	unsigned int chunk_lengths[desc->height];
    
	cl_mem pixel_buffer = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, px_len * sizeof(unsigned char), NULL, NULL);
    cl_mem bytes_buffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, max_size * sizeof(unsigned char), NULL, NULL);
    cl_mem chunk_lens_buffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, desc->height * sizeof(unsigned int), NULL, NULL);

    // TODO: adjust work item sizes in case img_height > CL_DEVICE_MAX_WORK_ITEM_SIZES
    clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), (void*)&pixel_buffer);
    clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), (void*)&bytes_buffer);
    clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), (void*)&chunk_lens_buffer);
    clSetKernelArg(ocl.kernel, 3, sizeof(int), (void*)&desc->width);
    clSetKernelArg(ocl.kernel, 4, sizeof(int), (void*)&desc->channels);

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #pragma GCC diagnostic ignored "-Wint-conversion"
    cl_command_queue command_queue = clCreateCommandQueue(ocl.context, ocl.device_id, NULL, NULL);
    #pragma GCC diagnostic pop

    clEnqueueWriteBuffer(
        command_queue,
        pixel_buffer,
        CL_FALSE,
        0,
        px_len * sizeof(unsigned char),
        pixels,
        0,
        NULL,
        NULL
    );
    
    clEnqueueNDRangeKernel(
        command_queue,
        ocl.kernel,
        1,
        NULL,
        &desc->height,
        NULL,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        bytes_buffer,
        CL_TRUE,
        0,
        max_size * sizeof(unsigned char),
        bytes,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        chunk_lens_buffer,
        CL_TRUE,
        0,
        desc->height * sizeof(unsigned int),
        chunk_lengths,
        0,
        NULL,
        NULL
    );

    // Release the resources
    clReleaseKernel(ocl.kernel);
    clReleaseProgram(ocl.program);
    clReleaseContext(ocl.context);
    clReleaseDevice(ocl.device_id);

    clReleaseMemObject(pixel_buffer);
    clReleaseMemObject(bytes_buffer);
    clReleaseMemObject(chunk_lens_buffer);
    

    // TODO: - add header info
    // 		 - merge compressed segments
    //		 - remove artifacts and redundant chunks

    int merged_size = 0;
    for (int i = 0; i < desc->height; i++){
        merged_size += chunk_lengths[i];
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
        memcpy(&merged[p], &bytes[k], chunk_lengths[i]);
        p += chunk_lengths[i];
    }

    // don't need segments anymore
    free(bytes);
    *out_len = merged_size;
        
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