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
            switchToDefaultDisplay();
            break;
        case 1:
            switchToContourCurves();
            break;
        case 2:
            switchToTriangleLines();
            break;
    }

}

void DisplayController::switchToContourCurves() {
    useContourCurves = true;
    useTriangleLines = false;
    mode = 1;
}

void DisplayController::switchToDefaultDisplay() {
    useContourCurves = false;
    useTriangleLines = false;
    mode = 0;
}

void DisplayController::switchToTriangleLines() {
    useContourCurves = false;
    useTriangleLines = true;
    mode = 2;
}

bool DisplayController::displayContourCurves() const
{
    return useContourCurves;
}

bool DisplayController::displayTriangles() const
{
    return useTriangleLines && !useContourCurves;
}
