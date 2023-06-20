//
// Created by major on 21-12-2021.
//

#ifndef TERRAINGEN_LIGHT_H
#define TERRAINGEN_LIGHT_H

#include <glm/glm.hpp>
#include "shader.h"

/**
 * Class to handle light to use in shaders with lights.
 */
class Light
{
private:
    glm::vec3 la;
    glm::vec3 ld;
    glm::vec3 ls;
    glm::vec3 position;
    float attenuation_c;
    float attenuation_l;
    float attenuation_q;

public:
    /**
     * Creates an instance of a light, setting the classic light illumination properties.
     * Uses a single 'ambiental' parameter to create a vector with the same value in the three position.
     * @param ambiental
     * @param diffuse
     * @param specular
     * @param position
     * @param constant_attenuation
     * @param linear_attenuation
     * @param quadratic_attenuation
     */
    Light(float ambiental, float diffuse, float specular, glm::vec3 position,
          float constant_attenuation, float linear_attenuation, float quadratic_attenuation);

    /**
     * Creates an instance of a light, setting the classic light illumination properties.
     * @param ambiental
     * @param diffuse
     * @param specular
     * @param position
     * @param constant_attenuation
     * @param linear_attenuation
     * @param quadratic_attenuation
     */
    Light(glm::vec3 ambiental, glm::vec3 diffuse, glm::vec3 specular, glm::vec3 position,
          float constant_attenuation, float linear_attenuation, float quadratic_attenuation);

    /**
     * Sets the uniforms of a lighting shader.
     *
     * Caution: Binding the shader is not this method's job
     * @param shader Shader to set the uniforms
     */
    void Bind(Shader &shader) const;

    /**
     * Sets the light position
     * @param pos
     */
    void setPosition(glm::vec3 pos)
    {
        position = pos;
    }
};

#endif //TERRAINGEN_LIGHT_H
