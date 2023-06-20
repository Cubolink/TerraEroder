//
// Created by major on 20-06-2023.
//
#include "color.h"

Color::HSV Color::rgb_to_hsv(RGB color) {
    float min = std::min(std::min(color.r, color.g), color.b);
    float max = std::max(std::max(color.r, color.g), color.b);
    if (min == max)
        return {0, 0, 0};

    float h, s, v;

    // H
    if (max == color.r)
    {
        if (color.g >= color.b)
            h = 60 * (color.g - color.b) / (max - min);
        else
            h = 60 * (color.g - color.b) / (max - min) + 360;
    } else if (max == color.g)
        h = 60 * (color.b - color.r) / (max - min) + 120;
    else
        h = 60 * (color.r - color.g) / (max - min) + 240;

    // S
    s = max == 0? 0: 1 - (min/max);

    // V
    v = max;

    return {h, s, v};
}

Color::RGB Color::hsv_to_rgb(Color::HSV color) {
    int h_i = ((int) (color.h / 60)) % 6;
    float f = (float) std::fmod(((float) h_i / 60), 6) - (float) h_i;
    float p = color.v * (1 - color.s);
    float q = color.v * (1 - f*color.s);
    float t = color.v * (1 - (1 - f) * color.s);

    switch (h_i) {
        case 0:
            return {color.v, t, p};
        case 1:
            return {q, color.v, p};
        case 2:
            return {p, color.v, t};
        case 3:
            return {p, q, color.v};
        case 4:
            return {t, p, color.v};
        case 5:
            return {color.v, p, q};
        default:
            return {0, 0, 0};

    }
}
