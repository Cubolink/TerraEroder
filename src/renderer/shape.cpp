#include "shape.h"


Shape::Shape(const Shape &shape)
: vertices(shape.vertices.begin(), shape.vertices.end()), indices(shape.indices.begin(), shape.indices.end()),
vbo(shape.vertices), vbl(shape.vbl), ibo(shape.indices) {
    vao.AddBuffer(vbo, vbl);
}


Shape::Shape(const std::vector<float>& vertices, const std::vector<unsigned int> &indices, const std::vector<int>& count_layouts)
: vertices(vertices.begin(), vertices.end()), indices(indices.begin(), indices.end()),
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
= default;

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
