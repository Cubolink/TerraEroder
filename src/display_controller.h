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

    bool showNormals;

public:
    DisplayController();

    void toggleDisplay();

    void switchToContourCurves();

    void switchToTriangleLines();

    void switchToDefaultDisplay();

    bool displayContourCurves() const;

    bool displayTriangles() const;

    void toggleNormals();

    bool displayNormals() const;
};


#endif //TERRAERODER_DISPLAY_CONTROLLER_H
