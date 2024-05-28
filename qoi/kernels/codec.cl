#define QOI_OP_INDEX  0x00 /* 00xxxxxx */
#define QOI_OP_DIFF   0x40 /* 01xxxxxx */
#define QOI_OP_LUMA   0x80 /* 10xxxxxx */
#define QOI_OP_RUN    0xc0 /* 11xxxxxx */
#define QOI_OP_RGB    0xfe /* 11111110 */
#define QOI_OP_RGBA   0xff /* 11111111 */

__kernel void encode(__global unsigned char *pixels, __global unsigned char *bytes, __global unsigned int *chunk_lens, int width, int channels)
{
	/*
	// grayscale test
	int id = get_global_id(0) * width * channels; // RGBA offset 4
    
	for (int i = 0; i < width*4; i+=4){
		bytes[id + i + 0] = 255 - pixels[id + i + 0];
        bytes[id + i + 1] = 255 - pixels[id + i + 1];
        bytes[id + i + 2] = 255 - pixels[id + i + 2];
        
        if (channel == 4) bytes[id + i + 3] = 255 - pixels[id + i + 3];
	}
	*/
	
	// encode pixels
	int id = get_global_id(0) * width * channels;
	// byte index, account for tags
	unsigned int p = get_global_id(0) * width * (channels + 1);
	unsigned int start = p;

	unsigned char index[64*4] = {0}; // qoi_rgba_t
	unsigned char px_r, px_g, px_b, px_a, px_prev_r, px_prev_g, px_prev_b, px_prev_a;

	int run = 0;
	px_r = px_prev_r = 0;
	px_g = px_prev_g = 0;
	px_b = px_prev_b = 0;
	px_a = px_prev_a = 255;
	
	
	for (int px_pos = 0; px_pos < width * channels; px_pos += channels){
		px_r = pixels[id + px_pos + 0];
		px_g = pixels[id + px_pos + 1];
		px_b = pixels[id + px_pos + 2];

		if (channels == 4) {
			px_a = pixels[id + px_pos + 3];
		}

		if (px_a == px_prev_a){
			bytes[p++] = QOI_OP_RGB;
			bytes[p++] = px_r;
			bytes[p++] = px_g;
			bytes[p++] = px_b;
		}
		else {
			bytes[p++] = QOI_OP_RGBA;
			bytes[p++] = px_r;
			bytes[p++] = px_g;
			bytes[p++] = px_b;
			bytes[p++] = px_a;
		}

		px_prev_r = px_r;
		px_prev_g = px_g;
		px_prev_b = px_b;
		px_prev_a = px_a;
	}
	chunk_lens[get_global_id(0)] = p - start;
}

