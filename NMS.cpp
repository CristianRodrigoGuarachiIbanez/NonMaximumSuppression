#include "NMS.h"


NMS::NMS(std::vector<cv::Rect> srcRects) : Rects(srcRects){
   
    sort_BB(Rects);
}
NMS::~NMS(){
    idxs.clear();
}

void NMS::sort_BB(const std::vector<cv::Rect>& srcRects){
    const size_t size = srcRects.size();
    if (!size){
        std::cerr << "Error\n";
        std::cerr << "Vector of Rects empty\n";
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box

    for (size_t i = 0; i < size; ++i)
    {
        this->idxs.emplace(srcRects[i].br().y, i);
    }

}
void NMS::calculateNMS( std::vector<cv::Rect>& resRects, float thresh, int neighbors = 0){
    resRects.clear();
    
    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = Rects[lastElem->second];

        int neigborsCount = 0;

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
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
            resRects.push_back(rect1);
    }

}


