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

void bilateralColorCPU(Mat padded, Mat out, int a, int b, int c, int fs, float* spat_filt, float den );

int main(int argc, char** argv)
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
    cout << "After type conversion : " << input_image.size() << "  " << input_image.type() << endl;
    

    float sig_spatial = atof(argv[3]);
    float sig_range = atof(argv[4]);
    float gauss_den1 = 2.0*sig_spatial*sig_spatial;
    float gauss_den2 = 2.0*sig_range*sig_range;
    struct timeval start, end;

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

   /* pad_image.convertTo(pad_image, CV_8UC3);
    imshow( "Padded image", pad_image);
    waitKey(2000); */

    //Creating output image...
    Mat output_image = input_image; //as of now, it's type is CV_32F (Type: 5)...

    //Implementing Direct Bilateral Filter...
    
    //Generating the spatial kernel as a vector...
    float* spatialFilter;
    spatialFilter = new float[filt_size*filt_size];
    for(int i = 0; i < filt_size; i++)
    {
        for(int j = 0; j < filt_size; j++)
        {
            spatialFilter[i * filt_size + j] = expf(-1*(( ( (i-center_pixel) * (i-center_pixel) ) + ( (j-center_pixel) * (j-center_pixel) )) / gauss_den1));
            //cout << i+1 << " " << j+1 << " " << spatialFilter[(i*filt_size) +j] << endl;
        }
    }   
    //Filtering...
    int m=input_image.rows;
    int n=input_image.cols;
    
    gettimeofday(&start, NULL);
    bilateralColorCPU(pad_image, output_image, m, n, center_pixel, filt_size, spatialFilter, gauss_den2);
    gettimeofday(&end, NULL);
    float elapsed_time = ((end.tv_sec - start.tv_sec)*1000) + ((end.tv_usec - start.tv_usec)/1000);
    cout << "Time elapsed for bilateral filtering is: " << elapsed_time << " ms"<< endl;
    
    output_image.convertTo(output_image, CV_8UC3);
    imshow("Output Image", output_image);
    waitKey(5000);
    vector<int> compression_params;
   	compression_params.push_back(IMWRITE_JPEG_PROGRESSIVE);
   	compression_params.push_back(9);
    imwrite(argv[2],output_image,compression_params); 
    return 0;
}

void bilateralColorCPU(Mat padded, Mat out, int a, int b, const int c, const int fs, float* spat_filt, float den )
{
    float normvalue,filtervalue,rangedist;
    cv::Vec3f pixelimg;
    for (int i = 0; i < a; i++)
    {
        cv::Vec3f* ptr = padded.ptr<cv::Vec3f>(i+c);
        cv::Vec3f* ptr_out = out.ptr<cv::Vec3f>(i);
        for (int j = 0; j < b; j++)
        { 
            int ind = j+c; 
            normvalue = 0;
            pixelimg = 0;    
            for (int k = 0; k < fs; k++)
            {
                cv::Vec3f* ptr1 = padded.ptr<cv::Vec3f>(i+k);
                for (int l = 0; l < fs; l++)
                {
                    //rangedist = saturate_cast<float> (norm(ptr[ind], ptr1[j+l]) * norm(ptr[ind], ptr1[j+l]));
                    rangedist = ((ptr[ind][0] - ptr1[j+l][0]) * (ptr[ind][0] - ptr1[j+l][0])) + ((ptr[ind][1] - ptr1[j+l][1]) * (ptr[ind][1] - ptr1[j+l][1])) + ((ptr[ind][2] - ptr1[j+l][2]) * (ptr[ind][2] - ptr1[j+l][2]));
                    filtervalue = spat_filt[(k*fs)+l] * expf(-rangedist/den); //product of spatial and range kernels...
                    normvalue += filtervalue; //normalization factor...
		            pixelimg += filtervalue * ptr1[j+l];
                }
            }
	     //out.at<float>(i,j) = pixelimg/normvalue;
         ptr_out[j] = pixelimg/normvalue; 
           //cout << "Writing at output image location (" << i+1 << "," << j+1 << ") a value of " << out.at<float>(i,j) << endl;
        }
    }
}
