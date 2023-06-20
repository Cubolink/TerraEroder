//
// Created by major on 19-06-2023.
//

#include "shape_factory.h"
#include "geometry.h"


Shape ShapeFactory::createTextureQuad(float tx0, float tx1, float ty0, float ty1)
{
    std::vector<float> vertices {
            -0.5f, -0.5f, 0.0f, tx0, ty0,
            0.5f, -0.5f, 0.0f, tx1, ty0,
            0.5f, 0.5f, 0.0f, tx1, ty1,
            -0.5f, 0.5f, 0.0f, tx0, ty1
    };
    std::vector<unsigned int> indices{
            0, 1, 2,
            2, 3, 0
    };
    std::vector<int> count_layouts;
    count_layouts.push_back(3);
    count_layouts.push_back(2);

    return {vertices, indices, count_layouts};
}

Shape ShapeFactory::createTextureQuad()
{
    return createTextureQuad(0, 1, 0, 1);
}

Shape ShapeFactory::createColorNormalCube(float r, float g, float b) {
    std::vector<float> vertices {
            // Z+
            -0.5, -0.5,  0.5, r, g, b,   0, 0, 1,
            0.5,  -0.5,  0.5, r, g, b,   0, 0, 1,
            0.5,   0.5,  0.5, r, g, b,   0, 0, 1,
            -0.5,  0.5,  0.5, r, g, b,   0, 0, 1,

            // Z-
            -0.5, -0.5, -0.5, r, g, b,   0, 0, -1,
            0.5,  -0.5, -0.5, r, g, b,   0, 0, -1,
            0.5,   0.5, -0.5, r, g, b,   0, 0, -1,
            -0.5,  0.5, -0.5, r, g, b,   0, 0, -1,

            // X+
            0.5, -0.5, -0.5, r, g, b,   1, 0, 0,
            0.5,  0.5, -0.5, r, g, b,   1, 0, 0,
            0.5,  0.5,  0.5, r, g, b,   1, 0, 0,
            0.5, -0.5,  0.5, r, g, b,   1, 0, 0,

            // X-
            -0.5, -0.5, -0.5, r, g, b,  -1, 0, 0,
            -0.5,  0.5, -0.5, r, g, b,  -1, 0, 0,
            -0.5,  0.5,  0.5, r, g, b,  -1, 0, 0,
            -0.5, -0.5,  0.5, r, g, b,  -1, 0, 0,

            // Y+
            -0.5, 0.5, -0.5, r, g, b,  0, 1, 0,
            0.5,  0.5, -0.5, r, g, b,  0, 1, 0,
            0.5,  0.5,  0.5, r, g, b,  0, 1, 0,
            -0.5, 0.5,  0.5, r, g, b,  0, 1, 0,

            // Y-
            -0.5, -0.5, -0.5, r, g, b,  0, -1, 0,
            0.5,  -0.5, -0.5, r, g, b,  0, -1, 0,
            0.5,  -0.5,  0.5, r, g, b,  0, -1, 0,
            -0.5, -0.5,  0.5, r, g, b,  0, -1, 0
    };
    std::vector<unsigned int> indices {
            0,  1,   2,  2,  3,  0,  // Z+
            7,  6,   5,  5,  4,  7,  // Z-
            8,  9,  10, 10, 11,  8,  // X+
            15, 14, 13, 13, 12, 15,  // X-
            19, 18, 17, 17, 16, 19,  // Y+
            20, 21, 22, 22, 23, 20  // Y-
    };
    std::vector<int> count_layouts;
    count_layouts.push_back(3);
    count_layouts.push_back(3);
    count_layouts.push_back(3);
    return {vertices, indices, count_layouts};
}

Shape ShapeFactory::createColorAxis(float length) {
    std::vector<float> vertices {
            -length, .0f, .0f, .0f, .0f, .0f, 1.f,
            length, .0f, .0f, 1.f, .0f, .0f, 1.f,

            .0f, -length, .0f, .0f, .0f, .0f, 1.f,
            .0f, length, .0f, .0f, .1f, .0f, 1.f,

            .0f, .0f, -length, .0f, .0f, .0f, 1.f,
            .0f, .0f, length, .0f, .0f, .1f, 1.f
    };
    std::vector<unsigned int> indices {
            0, 1,
            2, 3,
            4, 5
    };
    std::vector<int> count_layouts;
    count_layouts.push_back(3);
    count_layouts.push_back(4);

    return {vertices, indices, count_layouts};
}

Shape ShapeFactory::createColorNoiseMap(const std::vector<std::vector<float>>& map, float water_level = 0.45)
{
    long long int w = map.size();
    long long int h = map[0].size();
    std::vector<float> vertices(w * h * 9);
    float max_z = 0, min_z = 0;
    // Generate vertex positions
    for (long long int x = 0; x < w; x++)
    {
        for (long long int y = 0; y < h; y++)
        {
            vertices[9*(h*x + y)] = (float) x;
            vertices[9*(h*x + y) + 1] = (float) y;
            vertices[9*(h*x + y) + 2] = map[x][y];
            if (map[x][y]  > max_z)
                max_z = map[x][y];
            if (map[x][y] < min_z)
                min_z = map[x][y];
        }
    }
    // Generate vertex colores
    float range = max_z - min_z;
    float hue;
    for (long long int x = 0; x < w; x++)
    {
        for (long long int y = 0; y < h; y++)
        {
            if (map[x][y] < water_level)
            {
                vertices[9*(h*x + y) + 3] = 0;
                vertices[9*(h*x + y) + 4] = 0;
                vertices[9*(h*x + y) + 5] = 1;
            } else
            {
                hue = 240 * (map[x][y] - min_z) / range;
                Color::RGB color = Color::hsv_to_rgb({240 - hue, 1.f, 0.5f});
                vertices[9*(h*x + y) + 3] = color.r;
                vertices[9*(h*x + y) + 4] = color.g;
                vertices[9*(h*x + y) + 5] = color.b;
            }
        }
    }
    // Generate vertex normals
    for (long long int x = 0; x < w; x++)
    {
        for (long long int y = 0; y < h; y++)
        {
            long long int x_l = x - 1, x_r = x + 1;
            long long int y_l = y - 1, y_r = y + 1;

            std::vector<Geometry::Vector<float>> vectors;
            if (y_r < h)
                vectors.emplace_back(
                        vertices[9*(h*x + y_r) + 0] - vertices[9*(h*x + y) + 0],
                        vertices[9*(h*x + y_r) + 1] - vertices[9*(h*x + y) + 1],
                        vertices[9*(h*x + y_r) + 2] - vertices[9*(h*x + y) + 2]
                );
            if (x_l >= 0)
                vectors.emplace_back(
                        vertices[9*(h*x_l + y) + 0] - vertices[9*(h*x + y) + 0],
                        vertices[9*(h*x_l + y) + 1] - vertices[9*(h*x + y) + 1],
                        vertices[9*(h*x_l + y) + 2] - vertices[9*(h*x + y) + 2]
                );
            if (y_l >= 0)
                vectors.emplace_back(
                        vertices[9*(h*x + y_l) + 0] - vertices[9*(h*x + y) + 0],
                        vertices[9*(h*x + y_l) + 1] - vertices[9*(h*x + y) + 1],
                        vertices[9*(h*x + y_l) + 2] - vertices[9*(h*x + y) + 2]
                );
            if (x_r < w)
                vectors.emplace_back(
                        vertices[9*(h*x_r + y) + 0] - vertices[9*(h*x + y) + 0],
                        vertices[9*(h*x_r + y) + 1] - vertices[9*(h*x + y) + 1],
                        vertices[9*(h*x_r + y) + 2] - vertices[9*(h*x + y) + 2]);



            Geometry::Vector<float> v_sum(0, 0, 0);
            for (int i = 0; i < vectors.size(); i++)
                v_sum = v_sum + vectors[i].productoCruz(vectors[(i+1)%vectors.size()]);

            Geometry::Vector<double> normal = v_sum.normalizado();

            vertices[9*(h*x + y) + 6] = (float) normal.getX();
            vertices[9*(h*x + y) + 7] = (float) normal.getY();
            vertices[9*(h*x + y) + 8] = (float) normal.getZ();
        }
    }
    std::vector<unsigned int> indices;
    for (long long int x = 1; x < w; x++)
    {
        for (long long int y = 1; y < h; y++)
        {
            long long int i0 = h*(x-1) + (y-1);
            long long int i1 = h*x + (y-1);
            long long int i2 = h*x + y;
            long long int i3 = h*(x-1) + y;
            indices.push_back(i0);
            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i2);
            indices.push_back(i3);
            indices.push_back(i0);
        }
    }

    std::vector<int> count_layouts;
    count_layouts.push_back(3);
    count_layouts.push_back(3);
    count_layouts.push_back(3);

    return {vertices, indices, count_layouts};

}