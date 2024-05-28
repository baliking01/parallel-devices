#ifndef PARALLEL_QOI
#define PARALLEL_QOI

int parallel_qoi_write(const char operation, const char *filename, const void *data, const qoi_desc *desc);

#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include "kernel_loader.h"
#include "compact_types.h"


// encode target image using opencl parallel computing
// returns the size of the data written or 0 on failure
 int parallel_qoi_write(const char operation, const char *filename, const void *data, const qoi_desc *desc){
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
	int px_end = px_len - desc->channels;
	int channels = desc->channels;

	unsigned int chunk_lengths[desc->height];
    printf("px_len: %d\n", px_len);
	printf("width: %d\n", desc->width);
	printf("height: %d\n", desc->height);
	printf("channels: %d\n", channels);
	printf("max_size: %d\n", max_size);
    
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
    
    // TODO: - add header info
    // 		 - merge compressed segments
    //		 - remove artifacts and redundant chunks

    for (int i = 0; i < 200; i++){
    	printf("%d ", pixels[i]);
    }

    printf("\nConverted:\n");
    for (int i = 0; i < 200; i++){
        printf("%d ", bytes[i]);
    }

    printf("New\n");
    int offset;
    for (int i = 0; i < desc->height; i++){
        printf("Line: %d\n", i);
        offset = i * desc->width * (channels + 1);
        for (int j = 0; j < chunk_lengths[i];){
            switch (bytes[offset + j]){
                case QOI_OP_INDEX:

                case QOI_OP_DIFF:

                case QOI_OP_LUMA:

                case QOI_OP_RUN:

                case QOI_OP_RGB:
                    printf("R:%d G:%d B:%d\n", bytes[offset + j + 1], bytes[offset + j + 2], bytes[offset + j + 3]);
                    j += 4;
                    break;
                case QOI_OP_RGBA:
                    printf("R:%d G:%d B:%d A:%d\n", bytes[offset + j + 1], bytes[offset + j + 2], bytes[offset + j + 3], bytes[offset + j + 4]);
                    j += 5;
                    break;
                default:
                    break;
            }/*
            if (bytes[offset + j] == QOI_OP_RGB){
               printf("R:%d G:%d B:%d\n", bytes[offset + j + 1], bytes[offset + j + 2], bytes[offset + j + 3]);
               j += 4;
            }
            else if (bytes[offset + j] == QOI_OP_RGBA){
                printf("R:%d G:%d B:%d A:%d\n", bytes[offset + j + 1], bytes[offset + j + 2], bytes[offset + j + 3], bytes[offset + j + 4]);
                 j += 5;
            }*/
        }

    }

    printf("\n");
    for (int i = 0; i < desc->height; i++){
        printf("%d ", chunk_lengths[i]);
    }

    //int r = stbi_write_png("gray.png", desc->width, desc->height, channels, (void*)bytes, 0);


    // Release the resources
    clReleaseKernel(ocl.kernel);
    clReleaseProgram(ocl.program);
    clReleaseContext(ocl.context);
    clReleaseDevice(ocl.device_id);

    //free(host_buffer);
    free(pixel_buffer);
    free(bytes_buffer);

    printf("Success\n");
    exit(0);

    return 0;
}

#endif