//
// Created by Cubolink on 20-06-2023.
//

#include "obj_files.h"

#include <limits>

void Obj::storeShape(Shape &shape, const std::string &filepath) {
    std::ofstream f_stream(filepath);

    auto vertices = shape.getVertices();
    auto indices = shape.getIndices();

    for (unsigned int i = 0; i < vertices.size(); i += 9)
        f_stream << "v " << vertices[i] << " " << vertices[i+1] << " " << vertices[i+2] << "\n";
    for (unsigned int i = 6; i < vertices.size(); i += 9)
        f_stream << "vn " << vertices[i] << " " << vertices[i+1] << " " << vertices[i+2] << "\n";

    for (unsigned int i = 0; i < indices.size(); i += 3)
    {
        f_stream << "f ";
        f_stream << indices[i] + 1 << "//" << indices[i] + 1 << " ";
        f_stream << indices[i+1] + 1 << "//" << indices[i+1] + 1 << " ";
        f_stream << indices[i+2] + 1 << "//" << indices[i+2] + 1 << " ";
        f_stream << "\n";
    }
}

Shape Obj::readFile(const std::string &filepath) {
    std::ifstream fileStream(filepath);
    if (!fileStream) {
        std::cout << "Error reading the file " << filepath;
        exit(1);
    }

    std::vector<float> vertices;  // vx, vy, vz
    std::vector<float> normals;  // vnx, vny, vnz
    std::vector<float> colors;
    std::vector<unsigned int> indices;

    std::string line;
    while (std::getline(fileStream, line, '\n'))
    {
        if (line[0] == '#')
            continue;

        std::stringstream lineStringStream(line);
        std::string objElement;

        std::getline(lineStringStream, objElement, ' ');
        if (objElement == "v")
        {
            std::string vertexElement;

            int coordsCont = 0;
            while(std::getline(lineStringStream, vertexElement, ' '))
            {
                // x, y, z
                vertices.push_back(std::stof(vertexElement));
                coordsCont++;
            }
            if (coordsCont != 3)
                std::cout << "Unexpected non-tridimensional vertex" << std::endl;
        }
        else if (objElement == "vn")
        {
            std::string normalElement;

            int coordsCont = 0;
            while(std::getline(lineStringStream, normalElement, ' '))
            {
                normals.push_back(std::stof(normalElement));
                coordsCont++;
            }
            if (coordsCont != 3)
                std::cout << "Unexpected non-tridimensional normal" << std::endl;
        }
        else if (objElement == "f")
        {
            std::string faceElement;  // v_j/t_j/vn_j
            for (int j = 0; j < 3; j++){
                std::getline(lineStringStream, faceElement, ' ');
                std::stringstream faceElementStringStream0(faceElement);

                std::string faceSubElement;
                // v
                std::getline(faceElementStringStream0, faceSubElement, '/');
                indices.push_back(std::stoi(faceSubElement)-1);
                // t
                std::getline(faceElementStringStream0, faceSubElement, '/');  // skip texture index
                // vn
                std::getline(faceElementStringStream0, faceSubElement, '/');
                // skip normal index, we will assume is the same of the vertex
            }

        } else {
            std::cout << "Unhandled element in .obj file: '" << objElement << "'" << std::endl;
            continue;
        }
    }

    if (vertices.size() != normals.size())
    {
        std::cout << "[Error] Vertex-normal mismatch" << std::endl;
        exit(1);
    }

    float max_z = -std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    for (int i = 2; i < vertices.size(); i += 3)
    {
        if (vertices[i] > max_z)
            max_z = vertices[i];
        if (vertices[i] < min_z)
            min_z = vertices[i];
    }
    float z_range = max_z - min_z;
    float water_level = z_range/3 + min_z;

    std::vector<float> shape_vertices;  // 'vx, vy, vz, r, g, b, nvx, nvy, nvz' as a single vertex
    for (int i = 0; i < vertices.size(); i += 3)
    {
        // vertex position
        shape_vertices.push_back(vertices[i]);
        shape_vertices.push_back(vertices[i+1]);
        shape_vertices.push_back(vertices[i+2]);

        // vertex color
        if (vertices[i+2] < water_level)
        {
            // paint it blue
            shape_vertices.push_back(0);
            shape_vertices.push_back(0);
            shape_vertices.push_back(1);
        }
        else
        {
            float hue = 240 * (vertices[i+2] - min_z) / z_range;
            Color::RGB color = Color::hsv_to_rgb({240 - hue, 1.f, .5f});
            shape_vertices.push_back(color.r);
            shape_vertices.push_back(color.g);
            shape_vertices.push_back(color.b);
        }

        // vertex normal
        shape_vertices.push_back(normals[i]);
        shape_vertices.push_back(normals[i+1]);
        shape_vertices.push_back(normals[i+2]);
    }

    std::vector<int> count_layouts;
    count_layouts.push_back(3);
    count_layouts.push_back(3);
    count_layouts.push_back(3);

    std::cout << "End" << std::endl;
    return {shape_vertices, indices, count_layouts};
}