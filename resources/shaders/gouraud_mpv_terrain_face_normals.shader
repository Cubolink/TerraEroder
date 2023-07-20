#shader vertex
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 vertexNormal;

void main()
{
    //vec3 vertexPos = vec3(u_model * vec4(position, 1.0));
    gl_Position = vec4(position, 1.f);// u_projection * u_view * u_model * vec4(position, 1.0);
    vertexNormal = normal;
}

#shader geometry
#version 330 core

layout(triangles) in;
layout(line_strip, max_vertices=2) out;

in vec3 vertexNormal[];

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

void main() {
    vec4 faceCenter = vec4(0.f, 0.f, 0.f, 0.f);
    vec3 faceNormal = vec3(0.f, 0.f, 0.f);
    for (int i = 0; i < gl_in.length(); i++)
    {
        faceCenter += gl_in[i].gl_Position;
        faceNormal += vertexNormal[i];
    }
    faceCenter /= gl_in.length();
    faceCenter.w = 1.f;

    faceNormal = normalize(faceNormal);

    mat4 transformation = u_projection * u_view * u_model;

    gl_Position = transformation * faceCenter;
    EmitVertex();

    gl_Position = transformation * (faceCenter + 0.3f * vec4(faceNormal, 0.f));
    EmitVertex();

    EndPrimitive();
}

#shader fragment
#version 330 core

out vec4 fragColor;

void main()
{
    fragColor = vec4(0.f, 0.f, 0.f, 1.f);
}
