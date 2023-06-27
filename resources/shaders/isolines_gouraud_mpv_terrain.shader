#shader vertex
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec4 vertexColor;

uniform mat4 u_model;

uniform vec3 u_lightPosition;
uniform vec3 u_viewPosition;
uniform vec3 u_La;
uniform vec3 u_Ld;
uniform vec3 u_Ls;
uniform vec3 u_Ka;
uniform vec3 u_Kd;
uniform vec3 u_Ks;
uniform uint u_shininess;
uniform float u_constantAttenuation;
uniform float u_linearAttenuation;
uniform float u_quadraticAttenuation;

uniform float u_waterLevel;
uniform float u_deepestLevel;
uniform float u_levelRange;

int mod(int a, int b)
{
    return a - (b * int(a / b));
}

float fmod(float a, float b)
{
    return a - (b * int(a/b));
}

vec3 hsv_to_rgb(vec3 hsv_color) {
    // hsv coords <=> xyz coords
    int h_i = mod(int(hsv_color.x / 60), 6);
    float f = fmod(h_i/60.0, 6) - h_i;

    float p = hsv_color.z * (1 - hsv_color.y);
    float q = hsv_color.z * (1 - f*hsv_color.y);
    float t = hsv_color.z * (1 - (1 - f) * hsv_color.y);

    switch (h_i) {
        case 0:
        return vec3(hsv_color.z, t, p);
        case 1:
        return vec3(q, hsv_color.z, p);
        case 2:
        return vec3(p, hsv_color.z, t);
        case 3:
        return vec3(p, q, hsv_color.z);
        case 4:
        return vec3(t, p, hsv_color.z);
        case 5:
        return vec3(hsv_color.z, p, q);
        default:
        return vec3(0, 0, 0);
    }
}

void main()
{
    vec3 vertexPos = vec3(u_model * vec4(position, 1.0));  // illumination computation needs the u_model transformation
    gl_Position = vec4(position, 1.0);  // geometry shader will work with the original position

    // ambient
    vec3 ambient = u_Ka * u_La;

    // diffuse
    vec3 norm = normalize(normal);
    vec3 toLight = u_lightPosition - vertexPos;
    vec3 lightDir = normalize(toLight);
    vec3 diffuse = u_Kd * u_Ld * max(dot(norm, lightDir), 0.0);

    // specular
    vec3 viewDir = normalize(u_viewPosition - vertexPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shininess);
    vec3 specular = u_Ks * u_Ls * spec;

    // attenuation
    float distToLight = length(toLight);
    float attenuation = u_constantAttenuation
    + u_linearAttenuation * distToLight
    + u_quadraticAttenuation * distToLight * distToLight;

    // color
    vec3 color;
    if (position.z < u_waterLevel)
    {
        color = vec3(0, 0, 1);
    }
    else
    {
        float hue = 240 * (position.z - u_deepestLevel) / u_levelRange;
        color = hsv_to_rgb(vec3(240.f - hue, 1.f, .5f));
    }
    vec3 result = (ambient + ((diffuse + specular) / attenuation)) * color;
    vertexColor = vec4(result, 1.0);
}

#shader geometry
#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

layout(location = 0) in vec4 vertexColor[];

out vec4 interpolatedColor;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;


void main() {
    for (int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = u_projection * u_view * u_model * gl_in[i].gl_Position;
        interpolatedColor = vec4(vertexColor[i]);
        EmitVertex();
    }

    EndPrimitive();
}

#shader fragment
#version 330 core

in vec4 interpolatedColor;
out vec4 fragColor;

void main()
{
    fragColor = interpolatedColor;
}
