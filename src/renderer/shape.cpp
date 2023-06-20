#include "shape.h"


Shape::Shape(std::vector<float> vertices, std::vector<unsigned int> indices, const std::vector<int>& count_layouts)
: vertices(vertices), indices(indices),
  vbo(vertices),
  ibo(indices)
{
    for (int i: count_layouts)
    {
        vbl.Push<float>(i);  // ex: position coordinates layout, then color or texture, etc
    }
    vao.AddBuffer(vbo, vbl);
}

Shape::~Shape()
{

}

void Shape::Bind() const
{
    vao.Bind();
    vbo.Bind();
    ibo.Bind();
}

void Shape::Unbind() const
{
    vao.Unbind();
    vbo.Unbind();
    ibo.Unbind();
}

Shape &Shape::operator=(Shape shape) {

    if (this == &shape)
        return *this;

    vertices = shape.vertices;
    indices = shape.indices;

    vbo.updateData(vertices);
    ibo.updateData(indices);

    vbl = shape.vbl;
    vao.AddBuffer(vbo, vbl);

    return *this;
}
