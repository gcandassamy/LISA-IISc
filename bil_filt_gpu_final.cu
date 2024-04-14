#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<cuda.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<sys/time.h>

using namespace std;
using namespace cv;


__global__ void bilateralFilterGPU(cuda::PtrStepSz<float> padded, cuda::PtrStepSz<float> out, const int c, const int fs, float* spat_filt, float den)
{
    float normvalue,filtervalue,rangedist;
    float pixelimg;

    //Thread index calculations... 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((x < (out.cols)) && (y < (out.rows))) //precaution to avoid processing unnecessary threads...
    {
        float* optr = out.ptr(y);
        float* ptr = padded.ptr(y+c);
        int ind = x+c; 
        normvalue = 0;
        pixelimg = 0;  
            for (int k = 0; k < fs; k++)
            {
                float* ptr1 = padded.ptr(y+k);
                for (int l = 0; l < fs; l++)
                {
                    rangedist = (( ptr[ind] - ptr1[x+l] ) * ( ptr[ind] - ptr1[x+l] ) );
                    filtervalue = spat_filt[(k*fs)+l] * expf(-rangedist/den); //product of spatial and range kernels...
                    normvalue += filtervalue; //normalization factor...
                    pixelimg += (filtervalue * ptr1[x+l]);
                }
            }
        optr[x] = pixelimg/normvalue;
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
    cout << "Before type conversion : " << input_image.size() << "  " << input_image.type() << endl;
    
    input_image.convertTo(input_image,CV_32F);
    cout << "After type conversion : " << input_image.size << "  " << input_image.type() << endl;
    
    float sig_spatial = atof(argv[3]);
    float sig_range = atof(argv[4]);
    float gauss_den1 = 2.0*sig_spatial*sig_spatial;
    float gauss_den2 = 2.0*sig_range*sig_range;

    cout << "The values of spatial and range sigma are " << sig_spatial << " and " << sig_range << " respectively..."<< endl;
    cout << "The values of spatial and range gaussians' denominators are " << gauss_den1 << " and " << gauss_den2 << " respectively..."<< endl;

    const int filt_size = (int)ceil(6*sig_spatial + 1);
    const int center_pixel = (filt_size-1)/2;
    cout << "The value of filter size and center pixel are: " << filt_size << " and " << center_pixel << endl;

    //Creating zero padded image...
    Mat pad_image = Mat::zeros(input_image.rows+(2*center_pixel), input_image.cols+(2*center_pixel), input_image.type());
    cout << "Properties of padded image (before): " << pad_image.size() << " and " << pad_image.type() << endl;
    copyMakeBorder(input_image, pad_image, center_pixel, center_pixel, center_pixel, center_pixel, BORDER_CONSTANT);
    cout << "Properties of padded image (after): " << pad_image.size() << " and " << pad_image.type() << endl;

    //Creating output image...
    Mat output_image = input_image; //as of now, it's type is CV_32F (Type: 5)...

    //Implementing Direct Bilateral Filter...
    
    //Generating kernel as a vector attempt... (okay!)
    float* spatialFilter;
    spatialFilter = new float[filt_size*filt_size];
    float* d_spat_filt;
    for(int i = 0; i < filt_size; i++)
    {
        for(int j = 0; j < filt_size; j++)
        {
            spatialFilter[i * filt_size + j] = expf(-1*(( ( (i-center_pixel) * (i-center_pixel) ) + ( (j-center_pixel) * (j-center_pixel) )) / gauss_den1));
            //cout << i+1 << " " << j+1 << " " << spatialFilter[(i*filt_size) +j] << endl;
        }
    }
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

    //Declaring device copies of variables...
    cv::cuda::GpuMat padded, out;
    padded.upload(pad_image);
    out.upload(input_image);
    cout << "Creating GPU matrices..." << endl;
    cout << "GPU padded size: " << padded.size() << " and type: " << padded.type() << endl;
    cout << "GPU output size: " << out.size() << " and type: " << out.type() << endl;
    cout << out.cols << "\t" << out.rows << endl;
    
    //Invoking the kernel with suitable parameters...
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(input_image.rows/threadsPerBlock.x, input_image.cols/threadsPerBlock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    bilateralFilterGPU <<< numBlocks, threadsPerBlock >>> (padded, out, center_pixel, filt_size, d_spat_filt, gauss_den2);
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
    
    out.download(output_image);
    output_image.convertTo(output_image, CV_8U);
    imshow("Output Image", output_image);
    waitKey(2000);
    vector<int> compression_params;
   	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
   	compression_params.push_back(9);
    imwrite(argv[2],output_image,compression_params); 
    }

    catch (const cv::Exception& ex)
    {
        cout << " Exception occurred: " << ex.what() << endl;
    }
    
    return 0; 
}