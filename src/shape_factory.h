//
// Created by major on 19-06-2023.
//

#ifndef TERRAERODER_SHAPE_FACTORY_H
#define TERRAERODER_SHAPE_FACTORY_H

#include "shape.h"
#include "color.h"

/**
 * Class to handle shape creation.
 */
class ShapeFactory
{
public:

/**
 * Create a square shape with 3D vertices, and texture coordinates, in the XY plane
 *
 * @param tx0
 * @param tx1
 * @param ty0
 * @param ty1
 * @return
 */
    static Shape createTextureQuad(float tx0, float tx1, float ty0, float ty1);


/**
 * Creates a square shape with 3D vertices, and texture coordinates, in the XY plane
 *
 * @return
 */
    static Shape createTextureQuad();

/**
 * Creates a cube with 3D vertices, normals, and colors
 *
 * @param r
 * @param g
 * @param b
 * @return
 */
    static Shape createColorNormalCube(float r, float g, float b);

/**
 * Creates the XYZ colored axis, with 3D vertices, expected to be uses with GL_LINES instead of triangles
 *
 * @param length
 * @return
 */
    static Shape createColorAxis(float length);

/**
 * Creates a noise map, with 3D vertices, normals and colors. Uses the water_level to color some vertices with blue.
 *
 * @param map
 * @param water_level
 * @return
 */
    static Shape createColorNoiseMap(const std::vector<std::vector<float>>& map, float water_level);

};

#endif //TERRAERODER_SHAPE_FACTORY_H
