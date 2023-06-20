#shader vertex
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 v_color;

out vec4 pre_color;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

void main()
{
    vec4 pos4 = vec4(position, 1.0f);
    gl_Position = u_projection * u_view * u_model * pos4;
    pre_color = v_color;
}

#shader fragment
#version 330 core

in vec4 pre_color;
out vec4 color;

void main()
{
    color = pre_color;
}
