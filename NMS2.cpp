#include "NMS.h"
#include <assert.h>

NMS2::NMS2(std::vector<cv::Rect> srcRects, std::vector<float> scores, float score_thresh=0.f) : Rects(srcRects), SCORES(scores){   
    sort_BB_scores(SCORES, score_thresh);
}
NMS2::~NMS2(){
    idxs.clear();
}

void NMS2::sort_BB_scores(const std::vector<float>& scores, float score_thresh){
    const size_t size = scores.size();
    if (!size){
        std::cerr << "Error\n";
        std::cerr << "Vetor of Rects empty\n";
    }
    assert(Rects.size() == scores.size());
    
    if(score_thresh > 0.f){
        this->score_thresh = score_thresh;
        // Sort the bounding boxes by the detection score
        for (size_t i = 0; i < size; ++i){
		    if (scores[i] >= score_thresh)
			    this->idxs.emplace(scores[i], i);
            }
    }else{
        // Sort the bounding boxes by the detection score
        for (size_t i = 0; i < size; ++i){
            this->idxs.emplace(scores[i], i);
        }
    }
}

void NMS2::calculateNMS2( std::vector<cv::Rect>& resRects, float thresh, int neighbors = 0, float minScoresSum = 0.f){
    resRects.clear();
    
    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = Rects[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = Rects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {   
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum)
            resRects.push_back(rect1);
    }

}



void NMS2::calculateSoftNMS( std::vector<cv::Rect>& resRects, std::vector<float>& resScores, float iou_thresh, Methods method, float sigma = 0.5f){
    if (resRects.capacity() < idxs.size()){
		resRects.reserve(idxs.size());
		resScores.reserve(idxs.size());
	}

    // keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0){
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const cv::Rect& rect1 = Rects[lastElem->second];

		if (lastElem->first >= score_thresh){
			resRects.push_back(rect1);
			resScores.push_back(lastElem->first);
		}else{
			break;
		}

		idxs.erase(lastElem);

		for (auto pos = std::begin(idxs); pos != std::end(idxs); )	{
			// grab the current rectangle
			const cv::Rect& rect2 = Rects[pos->second];

			float intArea = static_cast<float>((rect1 & rect2).area());
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > iou_thresh){
				float weight = 1.f;
				switch (method)
				{
				case Methods::ClassicNMS:
					weight = 0;
					break;
				case Methods::LinearNMS:
					weight = 1.f - overlap;
					break;
				case Methods::GaussNMS:
					weight = exp(-(overlap * overlap) / sigma);
					break;
				}

				float newScore = pos->first * weight;
				if (newScore < score_thresh)
				{
					pos = idxs.erase(pos);
				}else{
					auto n = idxs.extract(pos);
					n.key() = newScore;
					idxs.insert(std::move(n));
					++pos;
				}
			}else{
				++pos;
			}
		}
	}
}



