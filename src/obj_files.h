//
// Created by Cubolink on 20-06-2023.
//

#ifndef TERRAINGEN_OBJ_FILES_H
#define TERRAINGEN_OBJ_FILES_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "shape.h"
#include "color.h"

namespace Obj
{
    /**
     * Stores a shape in a .obj file
     * @param shape
     * @param filepath
     */
    void storeShape(Shape &shape, const std::string &filepath);

    /**
     * Reads a .obj file and creates a shape
     * @param filepath
     */
    Shape readFile(const std::string &filepath);
}

#endif //TERRAINGEN_OBJ_FILES_H
