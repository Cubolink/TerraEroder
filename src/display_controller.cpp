//
// Created by Cubolink on 27-06-2023.
//

#include "display_controller.h"

DisplayController::DisplayController()
: useContourCurves(false), useTriangleLines(false), mode(0)
{

}

void DisplayController::toggleDisplay()
{
    mode++;
    switch (mode % 3) {
        case 0:
            useContourCurves = true;
            useTriangleLines = false;
            break;
        case 1:
            useContourCurves = false;
            useTriangleLines = false;
            break;
        case 2:
            useContourCurves = false;
            useTriangleLines = true;
            break;
    }

}

bool DisplayController::displayContourCurves() const
{
    return useContourCurves;
}

bool DisplayController::displayTriangles() const
{
    return useTriangleLines && !useContourCurves;
}
