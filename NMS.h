
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

/**
 * @brief nms
 * Non maximum suppression
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */

class NMS{
    public:
        NMS(std::vector<cv::Rect> srcRects);
        ~NMS();
        void calculateNMS( std::vector<cv::Rect>& resRects, float thresh, int neighbors);
    private:
        std::multimap<int, size_t> idxs;
        const std::vector<cv::Rect> Rects;
        void sort_BB(const  std::vector<cv::Rect>& srcRects);
        

};


enum class Methods
{
	ClassicNMS,
	LinearNMS,
	GaussNMS
};



/**
 * @brief nms2
 * Non maximum suppression with scores
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */

class NMS2{
    public:
        NMS2(std::vector<cv::Rect> srcRects, std::vector<float> scores, float score_thresh);
        ~NMS2();
        void calculateNMS2( std::vector<cv::Rect>& resRects, float thresh, int neighbors, float minScoresSum);
        void calculateSoftNMS( std::vector<cv::Rect>& resRects, std::vector<float>& resScores, float iou_thresh, Methods method, float sigma);
       
    private:
        std::multimap<float, size_t> idxs;
        const std::vector<cv::Rect> Rects;
        const std::vector<float>& SCORES;
        float score_thresh = 0.f;
        void sort_BB_scores(const std::vector<float>& scores, float score_thresh);
        

};

