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

__global__ void nlmColorGPU(cuda::PtrStepSz<float> pad0, cuda::PtrStepSz<float> pad1, cuda::PtrStepSz<float> pad2, cuda::PtrStepSz<float> out0, 
                            cuda::PtrStepSz<float> out1, cuda::PtrStepSz<float> out2,  const int fs, const int ps, const int t_pix, const int p_pix, float den)
{
    // Creating NLM filtered image... 
    float norm_value,filtervalue, frob_norm;
    float pixelimg1, pixelimg2, pixelimg3;

    //Thread index calculations... 
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if((a <out0.cols) && (b < out0.rows))
    {
        float* optr = out0.ptr(b);
        float* optr1 = out1.ptr(b);
        float* optr2 = out2.ptr(b);
        norm_value = 0;
        pixelimg1 = 0.f;
        pixelimg2 = 0.f;
        pixelimg3 = 0.f;    
        for (int k = 0; k < fs; k++)
        {
            float* ptr31 = pad0.ptr(b+p_pix+k);
            float* ptr32 = pad1.ptr(b+p_pix+k);
            float* ptr33 = pad2.ptr(b+p_pix+k);
            for (int l = 0; l < fs; l++)
            {
                frob_norm = 0.f;
                for(int x = 0; x < ps; x++)
                {
                    float* ptr11 = pad0.ptr(b+t_pix+x);
                    float* ptr12 = pad1.ptr(b+t_pix+x);
                    float* ptr13 = pad2.ptr(b+t_pix+x);
                    float* ptr21 = pad0.ptr(b+k+x);
                    float* ptr22 = pad1.ptr(b+k+x);
                    float* ptr23 = pad2.ptr(b+k+x);
                    for(int y = 0; y < ps; y++)
                    {
                        frob_norm +=  ((ptr11[a+t_pix+y] - ptr21[a+l+y]) * (ptr11[a+t_pix+y] - ptr21[a+l+y]))  +  ((ptr12[a+t_pix+y] - ptr22[a+l+y]) * 
                                    (ptr12[a+t_pix+y] - ptr22[a+l+y])) + ((ptr13[a+t_pix+y] - ptr23[a+l+y]) * (ptr13[a+t_pix+y] - ptr23[a+l+y])) ;
                    }
                }
                filtervalue = expf(-frob_norm/den); //filter coefficient...
                norm_value += filtervalue; //normalization factor...
                pixelimg1 += (filtervalue * ptr31[a+p_pix+l]);
                pixelimg2 += (filtervalue * ptr32[a+p_pix+l]);
                pixelimg3 += (filtervalue * ptr33[a+p_pix+l]);
            }
        }
        optr[a] = pixelimg1 / norm_value;
        optr1[a] = pixelimg2 / norm_value;
        optr2[a] = pixelimg3 / norm_value;
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
    Mat output_image(input_image); //as of now, it's type is CV_32F (Type: 5)...
    cout << "\nOutput image properties: " << output_image.rows << " x " << output_image.cols << " x " << output_image.channels() << " and " << output_image.type() << endl;

    //Implementing Non-Local Means Filter...
    
    //Declaring device copies of variables...
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
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(input_image.rows/threadsPerBlock.x, input_image.cols/threadsPerBlock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nlmColorGPU <<< numBlocks, threadsPerBlock >>> (padded[0], padded[1], padded[2], out[0], out[1], out[2],  box_filt_size, patch_size, tar_pix, patch_pix , gauss_den);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    } 
    
    float elapsed_time = 0.f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "\nTime elapsed for non-local means filtering using GPU is: " << elapsed_time << " ms" << endl;
    
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