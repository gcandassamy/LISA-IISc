#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<cuda.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/cudaarithm.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<sys/time.h>

using namespace std;
using namespace cv;


__global__ void bilateralColorGPU(cuda::PtrStepSz<float> pad0, cuda::PtrStepSz<float> pad1, cuda::PtrStepSz<float> pad2, cuda::PtrStepSz<float> out0,
                                  cuda::PtrStepSz<float> out1, cuda::PtrStepSz<float> out2, const int c, const int fs, float* spat_filt, float den)
{
    float normvalue,filtervalue,rangedist;
    float pixelimg1, pixelimg2, pixelimg3;

    //Thread index calculations... 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((x < out0.cols) && (y < out0.rows)) //precaution to avoid processing unnecessary threads...
    {
        float* optr = out0.ptr(y);
        float* optr1 = out1.ptr(y);
        float* optr2 = out2.ptr(y);
        float* ptr = pad0.ptr(y+c);
        float* ptr2 = pad1.ptr(y+c);
        float* ptr4 = pad2.ptr(y+c);
        int ind = x+c; 
        normvalue = 0;
        pixelimg1 = 0;
        pixelimg2 = 0;
        pixelimg3 = 0;  
            for (int k = 0; k < fs; k++)
            {
                float* ptr1 = pad0.ptr(y+k);
                float* ptr3 = pad1.ptr(y+k);
                float* ptr5 = pad2.ptr(y+k);
                for (int l = 0; l < fs; l++)
                {
                    rangedist = ((( ptr[ind] - ptr1[x+l] ) * ( ptr[ind] - ptr1[x+l] )) + (( ptr2[ind] - ptr3[x+l] ) * ( ptr2[ind] - ptr3[x+l] )) + (( ptr4[ind] - ptr5[x+l] ) * ( ptr4[ind] - ptr5[x+l] )) );
                    filtervalue = spat_filt[(k*fs)+l] * expf(-rangedist/den); //product of spatial and range kernels...
                    normvalue += filtervalue; //normalization factor...
                    pixelimg1 += (filtervalue * ptr1[x+l]);
                    pixelimg2 += (filtervalue * ptr3[x+l]);
                    pixelimg3 += (filtervalue * ptr5[x+l]);
                }
            }
        optr[x] = pixelimg1/normvalue;
        optr1[x] = pixelimg2/normvalue;
        optr2[x] = pixelimg3/normvalue;
    }
} 

int main(int argc, char** argv)
{
    try
    {
    if(argc != 5)
    {
        cout << "Expected format: <input_image> <output_image> <spatial_sigma> <range_sigma>" << endl;
        return -1;
    }
    Mat input_image = imread(argv[1], IMREAD_ANYCOLOR);
    
    if(!input_image.data) //just ensuring not empty...
    {
        cout << "No image data!";
        return -1;
    }
    imshow("Input Image", input_image);
    waitKey(1000);
    cout << "Before type conversion : " << input_image.rows << "  " << input_image.cols << "  " << input_image.channels() << "  " << input_image.type() << endl;
    
    input_image.convertTo(input_image,CV_32F);
    cout << "\nAfter type conversion : " << input_image.rows << "  " << input_image.cols << "  " << input_image.channels() << "  " << input_image.type() << endl;
    

    float sig_spatial = atof(argv[3]);
    float sig_range = atof(argv[4]);
    float gauss_den1 = 2.0*sig_spatial*sig_spatial;
    float gauss_den2 = 2.0*sig_range*sig_range;

    cout << "\nThe values of spatial and range sigma are " << sig_spatial << " and " << sig_range << " respectively..."<< endl;
    cout << "\nThe values of spatial and range gaussians' denominators are " << gauss_den1 << " and " << gauss_den2 << " respectively..."<< endl;

    const int filt_size = (int)ceil(6*sig_spatial + 1);
    const int center_pixel = (filt_size-1)/2;
    cout << "\nThe value of filter size and center pixel are: " << filt_size << " and " << center_pixel << endl;

    //Creating zero padded image...
    Mat pad_image = Mat::zeros(input_image.rows+(2*center_pixel), input_image.cols+(2*center_pixel), input_image.type());
    cout << "\nProperties of padded image (before): " << pad_image.size() << " x " << pad_image.channels() << " and " << pad_image.type() << endl;
    copyMakeBorder(input_image, pad_image, center_pixel, center_pixel, center_pixel, center_pixel, BORDER_CONSTANT);
    cout << "\nProperties of padded image (after): " << pad_image.size() << " x " << pad_image.channels() << " and " << pad_image.type() << endl;

    //Creating output image...   
    Mat output_image(input_image); //as of now, it's type is CV_32F (Type: 5)...
    cout << "\nOutput image properties: " << output_image.rows << " x " << output_image.cols << " x " << output_image.channels() << " and " << output_image.type() << endl;

    //Implementing Direct Bilateral Filter on GPU...

    //Generating the spatial kernel as a vector...
    float* spatialFilter;
    spatialFilter = new float[filt_size*filt_size];
    for(int i = 0; i < filt_size; i++)
    {
        for(int j = 0; j < filt_size; j++)
        {
            spatialFilter[i * filt_size + j] = expf(-1*(( ( (i-center_pixel) * (i-center_pixel) ) + ( (j-center_pixel) * (j-center_pixel) )) / gauss_den1));
        }
    }
    //Copying the kernel from host to device memory...
    float* d_spat_filt;   
    if(cudaMalloc((void**)&d_spat_filt, sizeof(float)*filt_size*filt_size) != cudaSuccess)
    {
        cout << "Nope! Could not allocate memory for spatial kernel..." << endl;
        return -1;
    }
    if(cudaMemcpy(d_spat_filt, spatialFilter, sizeof(float)*filt_size*filt_size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "Nope! Could not copy spatial filter into device memory..." << endl;
        return -1;
    }

    //Declaring device copies of Mat variables...
    cv::cuda::GpuMat padded[3], out[3];    
    cuda::split(pad_image, padded);
    cuda::split(input_image, out);
    cout << "\nCreating GPU matrices..." << endl;
    cout << "GPU padded[0] size: " << padded[0].rows << "x" << padded[0].cols << " x " << padded[0].channels() << " and type: " << padded[0].type() << endl;
    cout << "GPU padded[1] size: " << padded[1].rows << "x" << padded[1].cols << " x " << padded[1].channels() << " and type: " << padded[1].type() << endl;
    cout << "GPU padded[2] size: " << padded[2].rows << "x" << padded[2].cols << " x " << padded[2].channels() << " and type: " << padded[2].type() << endl;
    cout << "GPU output[0] size: " << out[0].rows << "x" << out[0].cols << " x " << out[0].channels() << " and type: " << out[0].type() << endl;
    cout << "GPU output[1] size: " << out[1].rows << "x" << out[1].cols << " x " << out[1].channels() << " and type: " << out[1].type() << endl;
    cout << "GPU output[2] size: " << out[2].rows << "x" << out[2].cols << " x " << out[2].channels() << " and type: " << out[2].type() << endl;
        
    //Invoking the kernel with suitable parameters...
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(input_image.rows/threadsPerBlock.x, input_image.cols/threadsPerBlock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    bilateralColorGPU <<< numBlocks, threadsPerBlock >>> (padded[0], padded[1], padded[2], out[0], out[1], out[2], center_pixel, filt_size, d_spat_filt, gauss_den2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    } 
    
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "Time elapsed for bilateral filtering is: " << elapsed_time << " milliseconds..."<< endl;
    
    cuda::merge(out, 3, output_image);
    output_image.convertTo(output_image, CV_8UC3);
    cout << "\nAfter filtering & converting, the properties of output are: " << output_image.rows << " x " << output_image.cols << " x " << output_image.channels() << " and type: " << output_image.type() << endl;
    imshow("Output Image", output_image);
    waitKey(3000);
    vector<int> compression_params;
   	compression_params.push_back(IMWRITE_JPEG_PROGRESSIVE);
   	compression_params.push_back(9);
    imwrite(argv[2],output_image,compression_params);  
    }

    catch (const cv::Exception& ex)
    {
        cout << " Exception occurred: " << ex.what() << endl;
    }
    return 0;
}
