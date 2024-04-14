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

__global__ void nlmFilterGPUGray(cuda::PtrStepSz<float> padded, cuda::PtrStepSz<float> out, const int fs, const int ps, const int t_pix, const int p_pix, float den)
{
    // Creating NLM filtered image... 
    float norm_value,filtervalue, pixelimg, frob_norm, diff_value;

    //Thread index calculations... 
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if((a <out.cols) && (b < out.rows))
    {
        float* ptr4 = out.ptr(b);
        norm_value = 0;
        pixelimg = 0;    
        for (int k = 0; k < fs; k++)
        {
            float* ptr3 = padded.ptr(b+p_pix+k);
            for (int l = 0; l < fs; l++)
            {
                frob_norm = 0.f;
                for(int x = 0; x < ps; x++)
                {
                    float* ptr1 = padded.ptr(b+t_pix+x);
                    float* ptr2 = padded.ptr(b+k+x);
                    for(int y = 0; y < ps; y++)
                    {
                        diff_value = ptr1[a+t_pix+y] - ptr2[a+l+y] ;
                        frob_norm +=  (diff_value * diff_value);
                    }
                }
                filtervalue = expf(-frob_norm/den); //filter coefficient...
                norm_value += filtervalue; //normalization factor...
		        pixelimg += (filtervalue * ptr3[a+p_pix+l]);
            }
        }
	    ptr4[a] = pixelimg/norm_value;
           //cout << "Writing at output image location (" << i+1 << "," << j+1 << ") a value of " << out.at<float>(i,j) << endl;
    }    
}

int main(int argc, char** argv)
{
    try
    {
    if(argc != 6)
    {
        cout << "Expected format: <input_image> <output_image> <search_radius> <patch_radius> <range_sigma>" << endl;
        return -1;
    }
    Mat input_image = imread(argv[1], IMREAD_GRAYSCALE);
    
    if(!input_image.data) //just ensuring not empty...
    {
        cout << "No image data!";
        return -1;
    }
    imshow("Input Image", input_image);
    waitKey(1000);
    cout << "Before type conversion : " << input_image.size() << "  " << input_image.type() << endl;
    
    input_image.convertTo(input_image,CV_32F);
    cout << "After type conversion : " << input_image.size() << "  " << input_image.type() << endl;

    //Reading the input parameters...
    const int search_rad = atof(argv[3]);
    const int patch_rad = atof(argv[4]);
    const float sig_range = atof(argv[5]);

    cout << "The values of search and patch radii are: " << search_rad << " and " << patch_rad << " respectively..." << endl;
    cout << "The value of range sigma is: " << sig_range << endl; 

    const int box_filt_size = (int)(ceil(2*search_rad + 1));
    const int patch_size = (int)ceil(2*patch_rad + 1);
    const float gauss_den = 2.0 * sig_range * sig_range;
    const int tar_pix = (box_filt_size - 1)/2;
    const int patch_pix = (patch_size - 1)/2;
    cout << "The value of range gaussian's denominator is: " << gauss_den << endl;
    cout << "The values of box filter's size and target pixel location are: " << box_filt_size << " and " << tar_pix << " respectively..." <<endl;
    cout << "The values of patch's size and pixel location are: " << patch_size << " and " << patch_pix << " respectively..." << endl;

    //Creating zero padded image...
    Mat pad_image = Mat::zeros(input_image.rows+(2*tar_pix)+(2*patch_pix), input_image.cols+(2*tar_pix)+(2*patch_pix), input_image.type());
    cout << "Properties of padded image (before): " << pad_image.size() << " and " << pad_image.type() << endl;
    copyMakeBorder(input_image, pad_image, tar_pix+patch_pix, tar_pix+patch_pix, tar_pix+patch_pix, tar_pix+patch_pix, BORDER_REFLECT101);
    cout << "Properties of padded image (after): " << pad_image.size() << " and " << pad_image.type() << endl;
   /* pad_image.convertTo(pad_image, CV_8U);
    imshow("Padded image", pad_image);
    waitKey(5000); */

    //Creating output image...
    Mat output_image = input_image; //as of now, it's type is CV_32F (Type: 5)...

    //Implementing Non-Local Means Filter...
    
    //Declaring device copies of variables...
    cv::cuda::GpuMat padded, out;
    padded.upload(pad_image);
    out.upload(input_image);
    cout << "Creating GPU matrices..." << endl;
    cout << "GPU padded size: " << padded.size() << " and type: " << padded.type() << endl;
    cout << "GPU output size: " << out.size() << " and type: " << out.type() << endl;
    cout << out.cols << "\t" << out.rows << endl;

    //Invoking the kernel with suitable parameters...
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(input_image.rows/threadsPerBlock.x, input_image.cols/threadsPerBlock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nlmFilterGPUGray <<< numBlocks, threadsPerBlock >>> (padded, out, box_filt_size, patch_size, tar_pix, patch_pix , gauss_den);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    } 
    
    float elapsed_time = 0.f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "Time elapsed for non-local means filtering is: " << elapsed_time << " ms" << endl;
    
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