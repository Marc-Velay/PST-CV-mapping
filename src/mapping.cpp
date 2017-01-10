#include "../headers/mapping.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace cv::ximgproc;

const String keys =
    "{help h usage ? |                  | print this message                                                }"
    "{@left          |../data/aloeL.jpg | left view of the stereopair                                       }"
    "{@right         |../data/aloeR.jpg | right view of the stereopair                                      }"
    "{GT             |../data/aloeGT.png| optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
    "{dst_path       |None              | optional path to save the resulting filtered disparity map        }"
    "{dst_raw_path   |None              | optional path to save raw disparity map before filtering          }"
    "{algorithm      |bm                | stereo matching method (bm or sgbm)                               }"
    "{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf)                     }"
    "{no-display     |                  | don't display results                                             }"
    "{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
    "{dst_conf_path  |None              | optional path to save the confidence map used in filtering        }"
    "{vis_mult       |1.0               | coefficient used to scale disparity map visualizations            }"
    "{max_disparity  |160               | parameter of stereo matching                                      }"
    "{window_size    |-1                | parameter of stereo matching                                      }"
    "{wls_lambda     |8000.0            | parameter of post-filtering                                       }"
    "{wls_sigma      |1.5               | parameter of post-filtering                                       }"
    ;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance) {
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}


inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}


static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
}





int main(int, char**)
{
    
    VideoCapture cap(0); // open the default camera (webcam)
    if(!cap.isOpened())  // check if we succeeded
        return -1;
        
    /*
    String filename = "vid/object.avi";
    VideoCapture cap(filename);
    if( !cap.isOpened() )
        throw "Error when reading steam_avi";
    */
    Mat edges;
    namedWindow("Original",1);
    namedWindow("edges",1);
    int length = int(cap.get(CV_CAP_PROP_FRAME_COUNT));
    int width = int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT)) +50;
    int wsize=0, max_disp = 64, GT_disp = 32;
    double lambda = 1.0, sigma=1.0, vis_mult=1.0;
    bool no_downscale = true, no_display = false, noGT = true;
    cout << "number of frames: " << length << endl;
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;
    String algo = "sgbm";
    String dst_path = "./output/";
    String dst_raw_path = "None";
    String dst_conf_path = "None";

    if(algo=="sgbm")
        wsize = 3; //default window size for SGBM
    else if(!no_downscale && algo=="bm")
        wsize = 7; //default window size for BM on downscaled views (downscaling is performed only for wls_conf)
    else
        wsize = 15; //default window size for BM on full-sized views


     Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(height, width,CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;

    Mat frame;
    Mat frameCpy;
    cap >> frame;
    GpuMat tempF;
    cuda::fastNlMeansDenoising(GpuMat(frame), tempF, 2, 7, 21);
    frame = cv::Mat(tempF);
    cvtColor(frame, frameCpy, COLOR_BGR2GRAY);	
    for(;;)
    {
        Mat secondFrame = frameCpy.clone(); //diffmap
        Mat left = frame.clone(); //stereo vision

        cap >> frame; // get a new frame from camera
        if(frame.empty()) break;
        Mat right = frame.clone();   //stereo vision

        /*******************************************************/
        left_for_matcher  = left.clone();
        right_for_matcher = right.clone();

        if(algo=="bm")
        {
            Ptr<StereoBM> matcher  = StereoBM::create(max_disp,wsize);
            matcher->setTextureThreshold(0);
            matcher->setUniquenessRatio(0);
            cvtColor(left_for_matcher,  left_for_matcher, COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else if(algo=="sgbm")
        {
            Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
            matcher->setUniquenessRatio(0);
            matcher->setDisp12MaxDiff(1000000);
            matcher->setSpeckleWindowSize(0);
            matcher->setP1(24*wsize*wsize);
            matcher->setP2(96*wsize*wsize);
            matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else
        {
            cout<<"Unsupported algorithm";
            return -1;
        }

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,left,filtered_disp,Mat(),ROI);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();


        //collect and print all the stats:
        cout.precision(2);
        cout<<"Matching time:  "<<matching_time<<"s"<<endl;
        cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
        cout<<endl;

        double MSE_before,percent_bad_before,MSE_after,percent_bad_after;
        if(!noGT)
        {
            MSE_before = computeMSE(GT_disp,left_disp,ROI);
            percent_bad_before = computeBadPixelPercent(GT_disp,left_disp,ROI);
            MSE_after = computeMSE(GT_disp,filtered_disp,ROI);
            percent_bad_after = computeBadPixelPercent(GT_disp,filtered_disp,ROI);

            cout.precision(5);
            cout<<"MSE before filtering: "<<MSE_before<<endl;
            cout<<"MSE after filtering:  "<<MSE_after<<endl;
            cout<<endl;
            cout.precision(3);
            cout<<"Percent of bad pixels before filtering: "<<percent_bad_before<<endl;
            cout<<"Percent of bad pixels after filtering:  "<<percent_bad_after<<endl;
        }

        /*if(dst_path!="None")
        {
            Mat filtered_disp_vis;
            getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
            imwrite(dst_path,filtered_disp_vis);
        }
        if(dst_raw_path!="None")
        {
            Mat raw_disp_vis;
            getDisparityVis(left_disp,raw_disp_vis,vis_mult);
            imwrite(dst_raw_path,raw_disp_vis);
        }
        if(dst_conf_path!="None")
        {
            imwrite(dst_conf_path,conf_map);
        }*/

        /*namedWindow("left", WINDOW_AUTOSIZE);
        imshow("left", left);
        namedWindow("right", WINDOW_AUTOSIZE);
        imshow("right", right);
        */
        /*if(!noGT)
        {
            Mat GT_disp_vis;
            getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
            namedWindow("ground-truth disparity", WINDOW_AUTOSIZE);
            imshow("ground-truth disparity", GT_disp_vis);
        }*/

        //! [visualization]
        Mat raw_disp_vis;
        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        namedWindow("raw disparity", WINDOW_AUTOSIZE);
        imshow("raw disparity", raw_disp_vis);
        moveWindow("raw disparity", 0, height);
        Mat filtered_disp_vis;
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        namedWindow("filtered disparity", WINDOW_AUTOSIZE);
        imshow("filtered disparity", filtered_disp_vis);
        moveWindow("filtered disparity", 30, height);
        //! [visualization]
        /*******************************************************/

        GpuMat tempF;
        cuda::fastNlMeansDenoising(GpuMat(frame), tempF, 2, 7, 21); //remove noise from image
        frame = cv::Mat(tempF);
        frameCpy = frame;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        cvtColor(frame, frameCpy, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);

        GpuMat d_frame0(frameCpy);
        GpuMat d_frame1(secondFrame);
        GpuMat d_flow(frameCpy.size(), CV_32FC2);

        //Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
        //Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
        Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
        //Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();
        /*
        {
            GpuMat d_frame0f;
            GpuMat d_frame1f;

            d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

            const int64 start = getTickCount();

            brox->calc(d_frame0f, d_frame1f, d_flow);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "Brox : " << timeSec << " sec" << endl;

            showFlow("Brox", d_flow);
            moveWindow("Brox",  width, height);
        }
        */
        /*
        {
            const int64 start = getTickCount();

            lk->calc(d_frame0, d_frame1, d_flow);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "LK : " << timeSec << " sec" << endl;

            showFlow("LK", d_flow);
            moveWindow("LK", width, height);
        }
        */
        
       {
            const int64 start = getTickCount();

            farn->calc(d_frame0, d_frame1, d_flow);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "Farn : " << timeSec << " sec" << endl;

            showFlow("Farn", d_flow);
            moveWindow("Farn", width, height);
        }
        
        /*
        {
            const int64 start = getTickCount();

            tvl1->calc(d_frame0, d_frame1, d_flow);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "TVL1 : " << timeSec << " sec" << endl;

            showFlow("TVL1", d_flow);
            moveWindow("TVL1", width, height);
        }
        */
        imshow("Original", frame);
        imshow("edges", edges);
        moveWindow("Original", 0, 0);
        moveWindow("edges", width, 0);
        if(waitKey(30) >= 0) break;
        cout << endl;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    cap.release();
    destroyAllWindows();
    return 0;
}