//
// Created by Cubolink on 27-06-2023.
//

#ifndef TERRAERODER_DISPLAY_CONTROLLER_H
#define TERRAERODER_DISPLAY_CONTROLLER_H


class DisplayController {
private:
    bool useContourCurves;
    bool useTriangleLines;
    unsigned int mode;

public:
    DisplayController();

    void toggleDisplay();

    bool displayContourCurves() const;

    bool displayTriangles() const;

};


#endif //TERRAERODER_DISPLAY_CONTROLLER_H
