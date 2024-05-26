__kernel void encode(__global unsigned char *px, __global unsigned char *byte, int width)
{
	int id = get_global_id(0) * width * 4; // RGBA offset 4
    
	for (int i = 0; i < width*4; i+=4){
		byte[id + i + 0] = 255 - px[id + i + 0];
        byte[id + i + 1] = 255 - px[id + i + 1];
        byte[id + i + 2] = 255 - px[id + i + 2];
        byte[id + i + 3] = 255 - px[id + i + 3];
	}


    
}

