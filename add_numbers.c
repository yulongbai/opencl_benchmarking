#define PROGRAM_FILE "add_numbers.cl"
#define KERNEL_FUNC "performNewIdeaIterationGPU"
#define ARRAY_SIZE (1<<14)

#include "ppm.h"

#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

inline void error(int assert, const char *msg) {
	if (!assert) {
		perror(msg);
		exit(EXIT_FAILURE);
	}
}

cl_int query_device(void) {
	cl_platform_id *platforms = NULL;
	char vendor_name[128] = { 0 };
	char platform_name[128] = { 0 };
	char platform_profile[128] = { 0 };
	cl_uint num_platforms = 0;

	cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
	if (CL_SUCCESS != err) {
		puts("clGetPlatformIDs");
		exit(-1);
	}
	platforms = (cl_platform_id *) malloc(
			sizeof(cl_platform_id) * num_platforms);
	if (NULL == platforms) {
		puts("malloc");
		exit(-1);
	}
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (CL_SUCCESS != err) {
		// handle error
		puts("clGetPlatformIDs 2");
		exit(-1);
	}
	for (cl_uint ui = 0; ui < num_platforms; ++ui) {
		err = clGetPlatformInfo(platforms[ui],
		CL_PLATFORM_VENDOR, 128 * sizeof(char), vendor_name,
		NULL);
		if (CL_SUCCESS != err) {
			// handle error
		}
		err = clGetPlatformInfo(platforms[ui],
		CL_PLATFORM_NAME, 128 * sizeof(char), platform_name,
		NULL);
		if (CL_SUCCESS != err) {
			// handle error
		}
		err = clGetPlatformInfo(platforms[ui],
		CL_PLATFORM_PROFILE, 128 * sizeof(char), platform_profile,
		NULL);
		printf("ui: %i, id: [%li], name: {%s}, vendor: {%s}, profile: {%s}\n",
				ui, (long)(platforms[ui]), platform_name, vendor_name,
				platform_profile);
	}
	return CL_SUCCESS;
}
/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*) malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1, (const char**) &program_buffer,
			&program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
				&log_size);
		program_log = (char*) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
				program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

typedef struct {
	float red, green, blue;
} AccuratePixel;

typedef struct {
	int x, y;
	AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage *convertImageToNewFormat(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *) malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*) malloc(
			image->x * image->y * sizeof(AccuratePixel));
	for (int i = 0; i < image->x * image->y; i++) {
		imageAccurate->data[i].red = (float) image->data[i].red;
		imageAccurate->data[i].green = (float) image->data[i].green;
		imageAccurate->data[i].blue = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}
PPMImage *accToPPM(AccurateImage *image) {
	// Make a copy
	PPMImage *imagePPM;
	imagePPM = (PPMImage *) malloc(sizeof(PPMImage));
	imagePPM->data = (PPMPixel*) malloc(
			image->x * image->y * sizeof(PPMPixel));
	for (int i = 0; i < image->x * image->y; i++) {
		imagePPM->data[i].red = image->data[i].red;
		imagePPM->data[i].green = image->data[i].green;
		imagePPM->data[i].blue = image->data[i].blue;
	}
	imagePPM->x = image->x;
	imagePPM->y = image->y;

	return imagePPM;
}
AccurateImage *createEmptyImage(PPMImage *image) {
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *) malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*) malloc(
			image->x * image->y * sizeof(AccuratePixel));
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

// free memory of an AccurateImage
void freeImage(AccurateImage *image) {
	free(image->data);
	free(image);
}

void error_check(cl_int err, const char *msg) {
	if (err < 0) {
		perror(msg);
		exit(1);
	};
}

void setBufferAndEnqueueKernel(cl_kernel *kernel,
		cl_mem* pixel_output_memobj, cl_mem* pixel_input_memobj,
		cl_command_queue* queue, size_t* global_size, size_t* local_size) {
	cl_int err = clSetKernelArg(*kernel, 0, sizeof(cl_mem), pixel_output_memobj);
	err |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), pixel_input_memobj);
	error_check(err, "Failed create kernel arguments");
	err = clEnqueueNDRangeKernel(*queue, *kernel, 2, NULL, global_size,
			local_size, 0, NULL, NULL);
	error_check(err, "Couldn't enqueue the kernel");
}

void new_idea(int size, cl_kernel* kernel, size_t global_size[],
		size_t local_size[], PPMImage* image, cl_mem* pixel_output_memobj,
		cl_mem* pixel_input_memobj, cl_command_queue* queue) {
	/* Create kernel arguments */
	cl_int err = clSetKernelArg(*kernel, 2, sizeof(int), &size);
	err |= clSetKernelArg(*kernel, 3, sizeof(int), &(image->x));
	err |= clSetKernelArg(*kernel, 4, sizeof(int), &(image->y));
	error_check(err, "Failed create kernel arguments");
	/* Enqueue kernel */
	setBufferAndEnqueueKernel(kernel, pixel_output_memobj,
			pixel_input_memobj, queue, global_size, local_size);
	setBufferAndEnqueueKernel(kernel, pixel_input_memobj,
			pixel_output_memobj, queue, global_size, local_size);
	setBufferAndEnqueueKernel(kernel, pixel_output_memobj,
			pixel_input_memobj, queue, global_size, local_size);
	setBufferAndEnqueueKernel(kernel, pixel_input_memobj,
			pixel_output_memobj, queue, global_size, local_size);
	setBufferAndEnqueueKernel(kernel, pixel_output_memobj,
			pixel_input_memobj, queue, global_size, local_size);
}

int main(int argc, char *argv[]) {
	assert(argc == 2);
	const char * kernel_name = argv[1];
	/* OpenCL structures */
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue queue;
	cl_int err;

	query_device();
	/* Create device and context */
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	/* Create a command queue, one queue per device */
	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Build program */
	program = build_program(context, device, kernel_name);

	/**
	 * Data preparation
	 */
	PPMImage *image = readPPM("flower.ppm");
	AccurateImage *imageUnchanged = convertImageToNewFormat(image); // save the unchanged image from input image
	AccurateImage *imageBuffer = createEmptyImage(image);

	/* Data and buffers */
	/* Create data buffer */
	size_t global_size[2] = {(size_t)image->x, (size_t)image->y};
	size_t local_size[2] = {16, 16};
	size_t pixels_size = image->x * image->y;
	cl_mem pixel_input_memobj = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE |	CL_MEM_COPY_HOST_PTR,
			pixels_size * sizeof(AccuratePixel),
			imageUnchanged->data,
			&err);
	error_check(err, "Couldn't create a buffer for pixel_input_memobj");
	cl_mem pixel_output_memobj = clCreateBuffer(
			context, CL_MEM_READ_WRITE,
			pixels_size * sizeof(AccuratePixel),
			NULL,
			&err);
	error_check(err, "Couldn't create a buffer for pixel_output_memobj");
	/* Create a kernel */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	error_check(err, "Couldn't create a kernel");

	/* Create kernel arguments */
	new_idea(3, &kernel, global_size, local_size, image, &pixel_output_memobj,
			&pixel_input_memobj, &queue);
	/* Read the kernel's output */
	err = clEnqueueReadBuffer(queue, pixel_output_memobj, CL_TRUE, 0, pixels_size*sizeof(AccuratePixel), imageBuffer->data,
			0, NULL, NULL);
	if (err < 0) {
		printf("err no: %d\n", err);
		perror("Couldn't read the buffer");
		exit(1);
	}
	/* Check result */
	PPMImage *imageOut = accToPPM(imageBuffer);
	writePPM("hello_flower.ppm", imageOut);
	free(imageOut->data);
	free(imageOut);
	/* Deallocate resources */
	freeImage(imageUnchanged);
	free(image->data);
	free(image);
	clReleaseKernel(kernel);
	clReleaseMemObject(pixel_output_memobj);
	clReleaseMemObject(pixel_input_memobj);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
