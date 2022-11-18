#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <fstream>

#ifdef cpp_lib
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif // cpp_lib

using namespace std;
using namespace cv;

static void help(const char** argv)
{
    cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
        "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
        "It's most known use is for faces.\n"
        "Usage:\n"
        << argv[0]
        << "   [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
        "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
        "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
        "   [--try-flip]\n"
        "   [filename|camera_index]\n\n"
        "example:\n"
        << argv[0]
        << " --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
        "During execution:\n\tHit any key to quit.\n"
        "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

vector<fs::path> loadFolderPhoto() { 
    const auto dir = "./image/";
    auto it = fs::directory_iterator(dir);
    vector<fs::path> array;

    //Массив ссылок на фото
    copy_if(fs::begin(it), fs::end(it), back_inserter(array),
        [](const auto& entry) {
            return fs::is_regular_file(entry);
        });

    return array;
}
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale, bool tryflip);

string cascadeName;
string nestedCascadeName;
int i = 0;


int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    double scale;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );                   

    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(samples::findFile(cascadeName)))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        //help(argv);
        return -1;
    }
    
    int control = 0;
    do {
        system("cls");
        cout << "Your welcome, this FacePhoto program" << endl << endl;
        cout << "1. Launch program" << endl;
        cout << "2. Close program" << endl;

        cin >> control;
        if (control == 1) {            
            
            for (auto& p : loadFolderPhoto())
            {                
                    Mat frame1 = imread(p.string());
                    cout << p.string();

                    detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip);
                    char c = (char)waitKey(3000);
            }
            destroyWindow("FacePhoto");
        }
    } while (control != 2);
    
    return 0;
}

void statistics(vector<Rect> faces) {

    ofstream out;          // поток для записи
    out.open("statistics.txt", ios_base::out | ios_base::app); // окрываем файл для записи

    if (out.is_open())
    {
        out << loadFolderPhoto()[i].string() << " количество найденных лиц -> " << faces.size() << std::endl;
        i++;
    }
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale, bool tryflip)
{
    double t = 0;
    vector<Rect> faces, faces2;
    
    //Закрашивание контуров в данные цвета
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    equalizeHist(smallImg, smallImg);
    t = (double)getTickCount();

    cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    if (tryflip)
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        
        for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;

    printf("detection time = %g ms\n", t * 1000 / getTickFrequency());

    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];                 
        Scalar color = colors[i % 8];        
        double aspect_ratio = (double)r.width / r.height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        {
            rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),
                Point(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
                color, 3, 8, 0);            
        }               
    }

    imshow("FacePhoto", img);     
    cout << "Count faces this photo = " << faces.size() << endl;
    statistics(faces);    
}