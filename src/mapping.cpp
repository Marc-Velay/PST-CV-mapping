#include "../headers/mapping.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;

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
    int counter=0;
    cout << "number of frames: " << length << endl;
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;


    Mat frame;
    Mat frameCpy;
    cap >> frame;
    GpuMat tempF;
    cuda::fastNlMeansDenoising(GpuMat(frame), tempF, 2, 7, 21);
    frame = cv::Mat(tempF);
    cvtColor(frame, frameCpy, COLOR_BGR2GRAY);	
    for(;;)
    {
        Mat secondFrame = frameCpy;
        cap >> frame; // get a new frame from camera
        if(frame.empty()) break;
        GpuMat tempF;
        cuda::fastNlMeansDenoising(GpuMat(frame), tempF, 2, 7, 21);
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
            moveWindow("Farn", width*2, 0);
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