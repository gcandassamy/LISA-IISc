#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<cuda.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<sys/time.h>

using namespace std;
using namespace cv;

void nlmColorCPU(Mat padded, Mat out, int a, int b, const int fs, const int ps, const int t_pix, const int p_pix, float den);

int main(int argc, char** argv)
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
    Mat output_image = input_image; //as of now, it's type is CV_32F (Type: 5)...

    //Implementing Non-Local Means Filter...

    int m = input_image.rows;
    int n = input_image.cols;    

    struct timeval start, end;
    gettimeofday(&start, NULL);
    nlmColorCPU(pad_image, output_image, m, n, box_filt_size, patch_size, tar_pix, patch_pix , gauss_den);
    gettimeofday(&end,NULL);
    float elapsed_time = ((end.tv_sec - start.tv_sec)*1000) + ((end.tv_usec - start.tv_usec)/1000);
    cout << "Time elapsed for non-local means filtering is: " << elapsed_time << " ms" << endl;
    
    output_image.convertTo(output_image, CV_8UC3);
    imshow("Output Image", output_image);
    waitKey(5000);
    vector<int> compression_params;
   	compression_params.push_back(IMWRITE_JPEG_PROGRESSIVE);
   	compression_params.push_back(9);
    imwrite(argv[2],output_image,compression_params); 
    return 0;
}

void nlmColorCPU(Mat padded, Mat out, int a, int b, const int fs, const int ps, const int t_pix, const int p_pix, float den)
{
    // Creating NLM filtered image... 
    float norm_value,filtervalue,rangedist, frob_norm;
    cv::Vec3f pixelimg;

    for (int i = 0; i < a; i++)
    {
        cv::Vec3f* ptr4 = out.ptr<cv::Vec3f>(i);
        for (int j = 0; j < b; j++)
        { 
            norm_value = 0;
            pixelimg = 0;    
            for (int k = 0; k < fs; k++)
            {
                cv::Vec3f* ptr3 = padded.ptr<cv::Vec3f>(i+p_pix+k);
                for (int l = 0; l < fs; l++)
                {
                    frob_norm = 0.f;
                    for(int x = 0; x < ps; x++)
                    {
                        cv::Vec3f* ptr1 = padded.ptr<cv::Vec3f>(i+t_pix+x);
                        cv::Vec3f* ptr2 = padded.ptr<cv::Vec3f>(i+k+x);
                        for(int y = 0; y < ps; y++)
                        {
                            frob_norm +=  ( (ptr1[j+t_pix+y][0] - ptr2[j+l+y][0]) * (ptr1[j+t_pix+y][0] - ptr2[j+l+y][0]) ) + ( (ptr1[j+t_pix+y][1] - ptr2[j+l+y][1]) * 
                                          (ptr1[j+t_pix+y][1] - ptr2[j+l+y][1]) )  + ( (ptr1[j+t_pix+y][2] - ptr2[j+l+y][2]) * (ptr1[j+t_pix+y][2] - ptr2[j+l+y][2]) );
                        }
                    }
                    filtervalue = expf(-frob_norm/den); //filter coefficient...
                    norm_value += filtervalue; //normalization factor...
		            pixelimg += (filtervalue * ptr3[j+p_pix+l]);
                }
            }
	     ptr4[j] = pixelimg/norm_value;
        }
    }
}