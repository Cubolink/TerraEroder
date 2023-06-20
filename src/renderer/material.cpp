//
// Created by major on 21-12-2021.
//

#include "material.h"

Material::Material(float ambiental, float diffuse, float specular, unsigned int shininess, Texture &texture)
: ka(ambiental, ambiental, ambiental),
  kd(diffuse, diffuse, diffuse),
  ks(specular, specular, specular),
  shininess(shininess),
  texture(texture)
{

}

void Material::Bind(Shader &shader) const
{
    texture.Bind();
    shader.SetUniform3f("u_Ka", ka.x, ka.y, ka.z);
    shader.SetUniform3f("u_Kd", kd.x, kd.y, kd.z);
    shader.SetUniform3f("u_Ks", ks.x, ks.y, ks.z);
    shader.SetUniform1ui("u_shininess", shininess);
}
