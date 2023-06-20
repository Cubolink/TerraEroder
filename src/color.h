//
// Created by major on 22-12-2021.
//

#ifndef TERRAINGEN_COLOR_H
#define TERRAINGEN_COLOR_H

#include <algorithm>
#include <cmath>

namespace Color
{
    struct RGB
    {
        float r;
        float g;
        float b;
    };

    struct HSV
    {
        float h;
        float s;
        float v;
    };

    /**
     * Takes a RGB color and returns the equivalent in HSV
     * @param color RGB color
     * @return HSV color
     */
    HSV rgb_to_hsv(RGB color);

    /**
     * Takes an HSV color and returns the equivalent in RGB
     * @param color HSV color
     * @return RGB color
     */
    RGB hsv_to_rgb(HSV color);

}


#endif //TERRAINGEN_COLOR_H
