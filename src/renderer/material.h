//
// Created by major on 21-12-2021.
//

#ifndef TERRAINGEN_MATERIAL_H
#define TERRAINGEN_MATERIAL_H

#include <glm/glm.hpp>
#include "texture.h"
#include "shader.h"


/**
 * Class to handle material shader properties, the reflection properties of a material.
 */
class Material
{
private:
    glm::vec3 ka;
    glm::vec3 kd;
    glm::vec3 ks;
    glm::uint shininess;
    Texture texture;

public:
    /**
     * Sets the different reflection properties of the material.
     * @param ambiental
     * @param diffuse
     * @param specular
     * @param shininess
     * @param texture
     */
    Material(float ambiental, float diffuse, float specular, unsigned int shininess, Texture &texture);

    /**
     * Caution: The shader has to be already bound
     */
    void Bind(Shader &shader) const;
};

#endif //TERRAINGEN_MATERIAL_H
