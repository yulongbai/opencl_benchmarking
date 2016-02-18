typedef struct {
	float red, green, blue;
} AccuratePixel;
__kernel
void performNewIdeaIterationGPU(
		__global AccuratePixel *pixelOut,
		__global AccuratePixel *pixelIn,
		int size,
		int len_x,
		int len_y) {
	
	int pos_y = get_global_id(1);
	int pos_x = get_global_id(0);
	// For each pixel we compute the magic number
	float sumR = 0;
	float sumG = 0;
	float sumB = 0;
	int countIncluded = 0;
	for(int y = -size; y <= size; y++) {
		for(int x = -size; x <= size; x++) {
			int currentX = pos_x + x;
			int currentY = pos_y + y;

			// Check if we are outside the bounds
			if(currentX < 0)
				continue;
			if(currentX >= len_x)
				continue;
			if(currentY < 0)
				continue;
			if(currentY >= len_y)
				continue;

			// Now we can begin
			int numberOfValuesInEachRow = len_x;
			int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
			sumR += pixelIn[offsetOfThePixel].red;
			sumG += pixelIn[offsetOfThePixel].green;
			sumB += pixelIn[offsetOfThePixel].blue;

			// Keep track of how many values we have included
			countIncluded++;
		}

	}

	// Now we compute the final value for all colours
	float recip = 1.0f/countIncluded;
	float valueR = sumR * recip;
	float valueG = sumG * recip;
	float valueB = sumB * recip;

	// Update the output image
	int offsetOfThePixel = (len_x * pos_y + pos_x);
	pixelOut[offsetOfThePixel].red = valueR;
	pixelOut[offsetOfThePixel].green = valueG;
	pixelOut[offsetOfThePixel].blue = valueB;

}
