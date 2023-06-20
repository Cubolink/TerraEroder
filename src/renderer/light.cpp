//
// Created by major on 21-12-2021.
//

#include "light.h"


Light::Light(float ambiental, float diffuse, float specular, glm::vec3 position, float constant_attenuation,
             float linear_attenuation, float quadratic_attenuation)
: la(ambiental, ambiental, ambiental),
  ld(diffuse, diffuse, diffuse),
  ls(specular, specular, specular),
  position(position),
  attenuation_c(constant_attenuation), attenuation_l(linear_attenuation), attenuation_q(quadratic_attenuation)
{

}

Light::Light(glm::vec3 ambiental, glm::vec3 diffuse, glm::vec3 specular, glm::vec3 position, float constant_attenuation,
             float linear_attenuation, float quadratic_attenuation)
 : la(ambiental), ld(diffuse), ls(specular), position(position),
   attenuation_c(constant_attenuation), attenuation_l(linear_attenuation), attenuation_q(quadratic_attenuation)
{

}

void Light::Bind(Shader &shader) const
{
    shader.SetUniform3f("u_La", la.x, la.y, la.z);
    shader.SetUniform3f("u_Ld", ld.x, ld.y, ld.z);
    shader.SetUniform3f("u_Ls", ls.x, ls.y, ls.z);
    shader.SetUniform3f("u_lightPosition", position.x, position.y, position.z);
    shader.SetUniform1f("u_constantAttenuation", attenuation_c);
    shader.SetUniform1f("u_linearAttenuation", attenuation_l);
    shader.SetUniform1f("u_quadraticAttenuation", attenuation_q);
}
