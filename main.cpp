#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv_modules.hpp>

using namespace std;

int main()
{
    cv::Mat src1, src2;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::BFMatcher matcher;
    std::string f;
    std::cout << "Vvedite deskriptor: ";
    std::cin >> f;
    if (f == "SURF"){
        detector = cv::xfeatures2d::SURF::create();
        extractor = detector;
    }
    else if (f == "SIFT"){
        detector = cv::xfeatures2d::SIFT::create();
        extractor = detector;
    }
    else if (f == "BRISK"){
        detector = cv::BRISK::create();
        extractor = detector;
    }
    cv::VideoCapture cap("/home/student/Desktop/sample_mpg.mov");
    if (!cap.isOpened())
        return -1;

    bool stop = false;
    int delay = 1000 / 60;
    bool firstfr = true;
    while (!stop) {
        bool result = cap.grab();
        if (result) {
            src1 = std::move(src2);
            cap >> src2;
            if (firstfr) {
                firstfr = false;
                continue;
            }
        }
        else {
            int key = cv::waitKey(150);
            if (key == 27){
                stop = true;
            }
            continue;
        }
        std::vector<cv::KeyPoint> keys1, keys2;
        detector->detect(src1, keys1);
        detector->detect(src2, keys2);
        cv::Mat descr1, descr2, img_matches;
        extractor->compute(src1, keys1, descr1);
        extractor->compute(src2, keys2, descr2);
        std::vector<cv::DMatch> matches, normmatches;
        matcher.match(descr1, descr2, matches);
        for (int i = 0; i < matches.size(); i++) {
            auto delta = keys2[matches[i].queryIdx].pt - keys1[matches[i].trainIdx].pt;
            if (delta.x * delta.x + delta.y * delta.y < 225) {
                normmatches.push_back(std::move(matches[i]));
            }
        }
        cv::drawMatches(src1, keys1, src2, keys2, normmatches, img_matches);
        cv::imshow("matches", img_matches);
        int key = cv::waitKey(delay);
        if (key == 27)
            stop = true;
    }
    return 0;
}

