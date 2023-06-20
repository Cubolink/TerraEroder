#shader vertex
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex_coord;

out vec2 v_tex_coord;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

void main()
{
    vec4 pos4 = vec4(position, 1.0f);
    gl_Position = u_projection * u_view * u_model * pos4;
    v_tex_coord = tex_coord;
}

#shader fragment
#version 330 core
in vec2 v_tex_coord;

layout(location = 0) out vec4 color;

uniform sampler2D u_texture;

void main()
{
    vec4 tex_color = texture(u_texture, v_tex_coord);
    color = tex_color;
}