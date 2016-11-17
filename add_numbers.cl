typedef struct {
	float red, green, blue;
} AccuratePixel;
__kernel
void performNewIdeaIterationGPU(
		__global AccuratePixel *pixels_out,
		__global AccuratePixel *pixels_in,
		int k_size,
		int len_x,
		int len_y) {
	
	int y = get_global_id(1);
	int x = get_global_id(0);
	float sum_r = 0;
	float sum_g = 0;
	float sum_b = 0;
	int normalized_divisor = 0;
	for (int offset_y = -k_size; offset_y <= k_size; ++offset_y) {
		int cur_y = y + offset_y;
		if (cur_y < 0 || cur_y >= len_y) continue;
		for (int offset_x = -k_size; offset_x <= k_size; ++offset_x) {
			int cur_x = x + offset_x;
			if (cur_x < 0 || cur_x >= len_x) continue;
		  int pixel_idx = cur_y*len_x + cur_x;

		  sum_r += pixels_in[pixel_idx].red;
		  sum_g += pixels_in[pixel_idx].green;
		  sum_b += pixels_in[pixel_idx].blue;
		  normalized_divisor++;
		}
	}
	int cur_idx = y*len_x + x;
	float recip = 1.0f/normalized_divisor;
	pixels_out[cur_idx].red = sum_r*recip;//regulate(sum_r*recip);
	pixels_out[cur_idx].green = sum_g*recip;//regulate(sum_g*recip);
	pixels_out[cur_idx].blue = sum_b*recip;//regulate(sum_b*recip);
}

