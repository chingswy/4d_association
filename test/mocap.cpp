#include "kruskal_associater.h"
#include "skel_updater.h"
#include "skel_painter.h"
#include "openpose.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <json/json.h>
#include "Timer.hpp"
// #define vis2d
#define write3d
std::string toOutPath(fs::path root, int frame, std::string posix){
    auto frameid = std::to_string(frame);
    return (root/fs::path(frameid + posix)).string();
}

int main()
{
    cv::FileStorage fs;
    fs.open("../config/config.yml", cv::FileStorage::READ);
    std::string setting_, dataPath, videoposix;
    fs["setting"] >> setting_;
    fs["path"] >> dataPath;
    fs["videoposix"] >> videoposix;

    std::cout << "experiment: " << setting_ << std::endl;
    std::cout << "path: " << dataPath << std::endl;

    auto setting = fs::path(setting_);
    auto outPath = fs::path("../output")/setting;
    fs::create_directories(outPath);
    auto detectPath = outPath/fs::path("detect");
    auto associPath = outPath/fs::path("associate");
    auto projecPath = outPath/fs::path("reproj");
    auto pointsPath = outPath/fs::path("keypoints");
    fs::create_directories(detectPath);
    fs::create_directories(associPath);
    fs::create_directories(projecPath);
    fs::create_directories(pointsPath);

    std::map<std::string, Camera> cameras = ParseCameras(dataPath + "calibration.json");
    Eigen::Matrix3Xf projs(3, cameras.size() * 4);
    std::vector<cv::Mat> rawImgs(cameras.size());
    std::vector<cv::VideoCapture> videos(cameras.size());
    std::vector<std::vector<OpenposeDetection>> seqDetections(cameras.size());
    auto skelType = SKEL15;
    const SkelDef& skelDef = GetSkelDef(skelType);

#pragma omp parallel for
    for (int i = 0; i < cameras.size(); i++) {
        auto iter = std::next(cameras.begin(), i);
        videos[i] = cv::VideoCapture(dataPath + "video/" + iter->first + videoposix);
        videos[i].set(cv::CAP_PROP_POS_FRAMES, 0);

        projs.middleCols(4 * i, 4) = iter->second.eiProj;
        seqDetections[i] = ParseDetections(dataPath + "detection/" + iter->first + ".txt");
        cv::Size imgSize(int(videos[i].get(cv::CAP_PROP_FRAME_WIDTH)), int(videos[i].get(cv::CAP_PROP_FRAME_HEIGHT)));
        for (auto&&detection : seqDetections[i]) {
            for (auto&& joints : detection.joints) {
                joints.row(0) *= imgSize.width;
                joints.row(1) *= imgSize.height;
            }
        }
        rawImgs[i] = cv::Mat();
    }
    KruskalAssociater associater(skelType, cameras);
    associater.SetMaxTempDist(0.3f);
    associater.SetMaxEpiDist(0.15f);
    associater.SetPlaneThetaWelsh(5e-3f);
    associater.SetEpiWeight(1.f);
    associater.SetTempWeight(2.f);
    associater.SetViewWeight(1.f);
    associater.SetPafWeight(1.f);
    associater.SetHierWeight(0.f);
    associater.SetViewCntWelsh(1.5);
    associater.SetMinCheckCnt(2);
    associater.SetNodeMultiplex(true);

    SkelFittingUpdater skelUpdater(skelType, "../data/skel/SKEL15");
    SkelPainter skelPainter(skelType);
    Timer timer;
    for (int frameIdx = 0; ; frameIdx++) {
        timer.tic();
#pragma omp parallel for
        for (int view = 0; view < cameras.size(); view++) {
            videos[view] >> rawImgs[view];
            if (rawImgs[view].empty()){
                std::cout << "empty view " << view << std::endl;
                return 0;
            }
            associater.SetDetection(view, seqDetections[view][frameIdx].Mapping(skelType));
        }
        timer.toc("load data");
        timer.tic();
        associater.SetSkels3dPrev(skelUpdater.GetSkel3d());
        associater.Associate();

        timer.toc("associate");
        timer.tic();
        skelUpdater.Update(associater.GetSkels2d(), projs);
        timer.toc(std::to_string(frameIdx));

#ifdef vis2d
        // save
        // const int layoutCols = cameras.size()%2==1?(cameras.size()+1)/2:cameras.size()/2;
        const int layoutCols = sqrt(cameras.size());
        cv::Mat detectImg, assocImg, reprojImg;
        std::vector<cv::Rect> rois = SkelPainter::MergeImgs(rawImgs, detectImg, layoutCols,
            { rawImgs.begin()->cols, rawImgs.begin()->rows});
        detectImg.copyTo(assocImg);
        detectImg.copyTo(reprojImg);
#pragma omp parallel for
        for (const auto& skel2d:associater.GetSkels2d()){
            cv::Mat tmp;
            detectImg.copyTo(tmp);
            for (int view = 0; view < cameras.size(); view++) {
                skelPainter.DrawAssoc(skel2d.second.middleCols(view * skelDef.jointSize, skelDef.jointSize), tmp(rois[view]), skel2d.first);
            }
            // cv::imwrite(pointsPath.string() + std::to_string(frameIdx) + "_" + std::to_string(skel2d.first) + ".jpg", tmp);
        }

#pragma omp parallel for
        for (int view = 0; view < cameras.size(); view++) {
            const OpenposeDetection detection = seqDetections[view][frameIdx].Mapping(skelType);
            skelPainter.DrawDetect(detection.joints, detection.pafs, detectImg(rois[view]));
            for (const auto& skel2d : associater.GetSkels2d())
                skelPainter.DrawAssoc(skel2d.second.middleCols(view * skelDef.jointSize, skelDef.jointSize), assocImg(rois[view]), skel2d.first);

            for(const auto& skel3d : skelUpdater.GetSkel3d())
                skelPainter.DrawReproj(skel3d.second, projs.middleCols(4 * view, 4), reprojImg(rois[view]), skel3d.first);
        }

        cv::imwrite(toOutPath(detectPath, frameIdx, ".jpg"), detectImg, 
            {cv::IMWRITE_JPEG_QUALITY, 50});
        cv::imwrite(toOutPath(associPath, frameIdx, ".jpg"), assocImg,
            {cv::IMWRITE_JPEG_QUALITY, 50});
        cv::imwrite(toOutPath(projecPath, frameIdx, ".jpg"), reprojImg,
            {cv::IMWRITE_JPEG_QUALITY, 50});
#endif

#ifdef write3d
		int nJoints = 15;
        std::cout << std::to_string(frameIdx) << std::endl;
        // 输出三维关键点
        std::ofstream resout(toOutPath(pointsPath, frameIdx, ".txt"));
        resout << skelUpdater.GetSkel3d().size() << std::endl;
        for(const auto& skel3d : skelUpdater.GetSkel3d()){
            auto trackId = skel3d.first;
            auto joints = skel3d.second.transpose();
            resout << joints.rows() << std::endl;
            resout << trackId << std::endl;
            resout << joints << std::endl;
        }
		resout.close();
        // 输出每个检测关节的指派结果
        std::ofstream asscocOut(toOutPath(pointsPath, frameIdx, "_assign.txt"));

        for(int view=0;view<cameras.size();view++){
            for(int nj=0;nj<15;nj++){
                asscocOut << view << " " << nj << " " << associater.m_assignMap[view][nj].rows() << std::endl;
                asscocOut << associater.m_assignMap[view][nj].transpose() << std::endl;
            }
        }
		asscocOut.close();

#endif

    }
    return 0;
}
