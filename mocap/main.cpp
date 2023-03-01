#include "kruskal_associater.h"
#include "skel_updater.h"
#include "skel_painter.h"
#include "openpose.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <json/json.h>

using ListStr = std::vector<std::string>;
using Str = std::string;

std::string toOutPath(fs::path root, int frame, std::string posix){
    auto frameid = std::to_string(frame);
    return (root/fs::path(frameid + posix)).string();
}

std::string num2string(const int value, const unsigned precision)
{
     std::ostringstream oss;
     oss << std::setw(precision) << std::setfill('0') << value;
     return oss.str();
}

int index(ListStr list, Str str){
	auto iter = std::find(list.begin(), list.end(), str);
	if(iter != list.end()){
		return std::distance(list.begin(), iter);
	}else{
		return -1;
	}
}

#define mylog(frame, x) std::cout << ">>> [" << frame << "] " << x << std::endl;

enum DataMode{
	VIDEO = 0,
	IMAGE = 1,
};

enum DatasetMode{
	ZJUMOCAP = 0,
	MHHI = 1,
	PANOPTIC = 2,
	CHI3D = 3,
	ZJUMOCAPv4 = 4,
	MHHI_SHAKE = 5,
	MHHI_JUMPX ,
	MHHI_0123,
	DATASET_SIZE,
};

int main(int argc, char *argv[])
{
	std::string dataset;
	DatasetMode data_mode;
	if(argc > 1) {
        dataset = std::string(argv[1]);
    } else {
        std::cout << ">>> Please specify the input path！" << std::endl;
        return 0;
    }
	std::string _data_mode = "";
	if(argc > 2){
		_data_mode = std::string(argv[2]);
		if(_data_mode == "zjumocap"){
			data_mode = DatasetMode::ZJUMOCAP;
		}else if(_data_mode == "chi3d"){
			data_mode = DatasetMode::CHI3D;
		}else if(_data_mode == "zjumocapv4"){
			data_mode = DatasetMode::ZJUMOCAPv4;
		}else if(_data_mode == "mhhi"){
			data_mode = DatasetMode::MHHI;
		}else if(_data_mode == "mhhi_shake"){
			data_mode = DatasetMode::MHHI_SHAKE;
		}else if(_data_mode == "mhhi_jumpx"){
			data_mode = DatasetMode::MHHI_JUMPX;
		}else if(_data_mode == "mhhi0123"){
			data_mode = DatasetMode::MHHI_0123;
		}else{
			exit(0);
		}
        std::cout << ">>> Set data mode to " << data_mode << " given " << _data_mode  << std::endl;
	}else{
		data_mode = DatasetMode::CHI3D;
	}
	DataMode mode = IMAGE;
	std::string IMAGE_EXT = ".jpg";

	std::vector<std::string> _camlist, _camvis;
	int start_frame = 0;
	if(data_mode == DatasetMode::MHHI){
		_camlist = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"};
		_camvis = {"00", "04", "10", "11"};
		IMAGE_EXT = ".png";
	}
	else if(data_mode == DatasetMode::MHHI_SHAKE){
		_camlist = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"};
		_camvis = {"00", "04", "10", "02"};
		IMAGE_EXT = ".png";
		start_frame = 151;
	}
	else if(data_mode == DatasetMode::MHHI_JUMPX){
		_camlist = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "10", "11"};
		_camvis = {"00", "04", "10", "02"};
		IMAGE_EXT = ".png";
	}
	else if(data_mode == DatasetMode::MHHI_0123){
		_camlist = {"00", "01", "02", "03"};
		_camvis = {"00", "01", "02", "03"};
		IMAGE_EXT = ".png";
	}
	else if(data_mode == DatasetMode::ZJUMOCAP){
		// _camlist = {"01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"};
		_camlist = {"01", "03", "05", "07", "09", "11", "13", "15", "17", "19", "21", "23"};
		_camvis = {"01", "07", "13", "19"};
		IMAGE_EXT = ".jpg";
	}
	else if(data_mode == DatasetMode::ZJUMOCAPv4){
		_camlist = {"01", "07", "13", "19"};
		_camvis = {"01", "07", "13", "19"};
		IMAGE_EXT = ".jpg";
	}
	else if(data_mode == DatasetMode::CHI3D){
		_camlist = {"50591643", "58860488", "60457274", "65906101"};
		_camvis = {"50591643", "58860488", "60457274", "65906101"};
	}
	std::vector<std::string> camlist = _camlist;
	std::vector<std::string> camvis = _camvis;
	std::map<std::string, Camera> cameras_all = ParseCameras(dataset + "/calibration.json");
	// 只保留部分cameras
	std::map<std::string, Camera> cameras;
	for(auto cam: camlist){
		cameras[cam] = cameras_all[cam];
	}
	auto inp_path = fs::path(dataset);
	auto outroot = inp_path/fs::path("association_"+_data_mode);
	std::vector<std::string> outlist = {"detect", "associate", "reproj", "keypoints"};
    std::unordered_map<std::string, fs::path> output_path;
    for(auto name: outlist){
        output_path[name] = outroot/fs::path(name);
        fs::create_directories(output_path[name]);
    }

	Eigen::Matrix3Xf projs(3, cameras.size() * 4);
	std::vector<cv::Mat> rawImgs(camvis.size());
	std::vector<cv::VideoCapture> videos(cameras.size());
	std::vector<fs::path> images_path(cameras.size());
	std::vector<std::vector<OpenposeDetection>> seqDetections(cameras.size());
	const SkelDef& skelDef = GetSkelDef(SKEL15);
	std::vector<std::map<int, Eigen::Matrix4Xf>> skels;

#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++) {
		auto iter = std::next(cameras.begin(), i);
		cv::Size imgSize;
		if(mode == DataMode::VIDEO){
			videos[i] = cv::VideoCapture(dataset + "/video/" + iter->first + ".mp4");
			videos[i].set(cv::CAP_PROP_POS_FRAMES, 0);
			imgSize = cv::Size(int(videos[i].get(cv::CAP_PROP_FRAME_WIDTH)), int(videos[i].get(cv::CAP_PROP_FRAME_HEIGHT)));
		}else{
			// reading the images
			images_path[i] = inp_path/fs::path("images")/fs::path(iter->first);
			std::string imgpath = images_path[i]/(num2string(start_frame, 6)+IMAGE_EXT);
			cv::Mat img = cv::imread(imgpath);
			imgSize = img.size();
			if(imgSize.height == 0){
				std::cout << "Fail to load image " << imgpath << std::endl;
			}
		}

		std::cout << "view: " << iter->first << " imgSize: " << imgSize << std::endl;
		projs.middleCols(4 * i, 4) = iter->second.eiProj;
		seqDetections[i] = ParseDetections(dataset + "/detection/" + iter->first + ".txt");
		std::cout << "view: " << iter->first << " detections: " << seqDetections[i].size() << std::endl;
		for (auto&& detection : seqDetections[i]) {
			for (auto&& joints : detection.joints) {
				joints.row(0) *= (imgSize.width - 1);
				joints.row(1) *= (imgSize.height - 1);
			}
		}
		if(index(camvis, iter->first) >= 0){
			std::cout << "view: " << iter->first << " imgSize: " << index(camvis, iter->first) << std::endl;
			rawImgs[index(camvis, iter->first)].create(imgSize, CV_8UC3);
		}
	}

	KruskalAssociater associater(SKEL15, cameras);
	associater.SetMaxTempDist(0.3f);
	associater.SetMaxEpiDist(0.15f);
	associater.SetEpiWeight(1.f);
	associater.SetTempWeight(2.f);
	associater.SetViewWeight(1.f);
	associater.SetPafWeight(2.f);
	associater.SetHierWeight(1.f);
	associater.SetViewCntWelsh(1.0);
	associater.SetMinCheckCnt(10);
	associater.SetNodeMultiplex(true);
	associater.SetNormalizeEdge(true);			// new feature

	SkelPainter skelPainter(SKEL15);
	skelPainter.rate = 512.f / float(cameras.begin()->second.imgSize.width);
	// SkelFittingUpdater skelUpdater(SKEL19, "../data/skel/SKEL19");
	SkelFittingUpdater skelUpdater(SKEL15, "../data/skel/SKEL15");
	skelUpdater.SetTemporalTransTerm(1e-1f / std::pow(skelPainter.rate, 2));
	skelUpdater.SetTemporalPoseTerm(1e-1f / std::pow(skelPainter.rate, 2));
	cv::Mat detectImg, assocImg, reprojImg;
	cv::Mat resizeImg;
	for (int frameIdx = start_frame; ; frameIdx++) {
		bool flag = true;
		mylog(frameIdx, "Reading images");
		for (int view = 0; view < cameras.size(); view++) {
			if(frameIdx - start_frame >= seqDetections[view].size()){
				flag = false;
				break;
			}
			associater.SetDetection(view, seqDetections[view][frameIdx-start_frame].Mapping(SKEL15));
			if(index(camvis, camlist[view]) < 0){
				continue;
			}
			int _view = index(camvis, camlist[view]);
			if(mode == DataMode::VIDEO){
				videos[view] >> rawImgs[_view];
			}else{
				auto imgname = (images_path[view]/fs::path(num2string(frameIdx, 6) + IMAGE_EXT)).string();
				// need to visualize
				rawImgs[_view] = cv::imread(imgname);
			}
			if (rawImgs[_view].empty()) {
				std::cout << "frameIdx: " << frameIdx << " view: " << view << " empty" << std::endl;
				flag = false;
				break;
			}
			cv::resize(rawImgs[_view], rawImgs[_view], cv::Size(), skelPainter.rate, skelPainter.rate);
		}
		if (!flag)
			break;
		mylog(frameIdx, "Associate");
		associater.SetSkels3dPrev(skelUpdater.GetSkel3d());
		associater.Associate();
		skelUpdater.Update(associater.GetSkels2d(), projs);

		mylog(frameIdx, "Visualize");
		// save
		const int layoutCols = std::sqrt(camvis.size()) + 0.5;
		std::vector<cv::Rect> rois = SkelPainter::MergeImgs(rawImgs, detectImg, layoutCols,
			{ rawImgs.begin()->cols, rawImgs.begin()->rows});
		detectImg.copyTo(assocImg);
		detectImg.copyTo(reprojImg);

#pragma omp parallel for
		for (int _view = 0; _view < camvis.size(); _view++) {
			int view = index(camlist, camvis[_view]);
			const OpenposeDetection detection = seqDetections[view][frameIdx-start_frame].Mapping(SKEL15);
			skelPainter.DrawDetect(detection.joints, detection.pafs, detectImg(rois[_view]));
			for (const auto& skel2d : associater.GetSkels2d())
				skelPainter.DrawAssoc(skel2d.second.middleCols(view * skelDef.jointSize, skelDef.jointSize), assocImg(rois[_view]), skel2d.first);

			for(const auto& skel3d : skelUpdater.GetSkel3d())
				skelPainter.DrawReproj(skel3d.second, projs.middleCols(4 * view, 4), reprojImg(rois[_view]), skel3d.first);
		}

		skels.emplace_back(skelUpdater.GetSkel3d());
		auto skel3d = skelUpdater.GetSkel3d();
		auto outname = toOutPath(output_path["keypoints"], frameIdx, ".txt");
		std::ofstream outfile(outname);
		outfile << skel3d.size() << std::endl;
		for(const auto& skel: skel3d){
			outfile << skel.first << std::endl;
			outfile << skel.second.transpose() << std::endl;
		}
		outfile.close();

		cv::imwrite(toOutPath(output_path["detect"], frameIdx, ".jpg"), detectImg);
		cv::imwrite(toOutPath(output_path["associate"], frameIdx, ".jpg"), assocImg);
		cv::imwrite(toOutPath(output_path["reproj"], frameIdx, ".jpg"), reprojImg);
		std::cout << std::to_string(frameIdx) << std::endl;
	}
	skels.clear();
	// SerializeSkels(skels, "../output/skel.txt");
	return 0;
}
