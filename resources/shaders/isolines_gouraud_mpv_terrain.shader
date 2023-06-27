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
layout(line_strip, max_vertices=40) out;

layout(location = 0) in vec4 vertexColor[];

out vec4 interpolatedColor;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_deepestLevel;
uniform float u_levelRange;

vec4 interpolate(vec4 A, vec4 B, float lambda)
{
    return A + lambda*(B - A);
}

#define A gl_in[0].gl_Position
#define B gl_in[1].gl_Position
#define C gl_in[2].gl_Position
#define colorA vec4(vertexColor[0])
#define colorB vec4(vertexColor[1])
#define colorC vec4(vertexColor[2])


void main() {

    int u_nIsolines = 20;
    for (int kIsoline = 0; kIsoline < u_nIsolines; kIsoline++)
    {
        // Find the two points where the triangle intersect the isoline, if any.
        float isolineZLevel = u_deepestLevel + u_levelRange * float(kIsoline) / u_nIsolines;

        // Check if AB, BC or CA are parallel to the isolineZLevel. If true, both extrems will form the primitive
        if ((B.z - A.z == 0) && A.z == isolineZLevel)
        {
            gl_Position = u_projection * u_view * u_model * A; interpolatedColor = colorA;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * B; interpolatedColor = colorB;
            EmitVertex();
            EndPrimitive();
            continue;
        } else if ((C.z - B.z == 0) && B.z == isolineZLevel)
        {
            gl_Position = u_projection * u_view * u_model * B; interpolatedColor = colorB;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * C; interpolatedColor = colorC;
            EmitVertex();
            EndPrimitive();
            continue;
        } else if ((A.z - C.z == 0 && C.z == isolineZLevel))
        {
            gl_Position = u_projection * u_view * u_model * C; interpolatedColor = colorC;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * A; interpolatedColor = colorA;
            EmitVertex();
            EndPrimitive();
            continue;
        }
        // Check intersections

        int nVerticesFound = 0;
        // Check the intersection for AB
        {
            float lambda = (isolineZLevel - A.z) / (B.z - A.z);
            if (0.f <= lambda && lambda <= 1.f)
            {
                gl_Position = u_projection * u_view * u_model * interpolate(A, B, lambda);
                interpolatedColor = interpolate(colorA, colorB, lambda);
                EmitVertex();
                nVerticesFound++;
            }
        }
        // Check the intersection for BC
        {
            float lambda = (isolineZLevel - B.z) / (C.z - B.z);
            if (0.f <= lambda && lambda <= 1.f)
            {
                gl_Position = u_projection * u_view * u_model * interpolate(B, C, lambda);
                interpolatedColor = interpolate(colorB, colorC, lambda);
                EmitVertex();
                nVerticesFound++;
            }
        }
        // Check the intersection for CA
        {
            float lambda = (isolineZLevel - C.z) / (A.z - C.z);
            if (0.f <= lambda && lambda <= 1.f)
            {
                gl_Position = u_projection * u_view * u_model * interpolate(C, A, lambda);
                interpolatedColor = interpolate(colorC, colorA, lambda);
                EmitVertex();
                nVerticesFound++;
            }
        }
        if (nVerticesFound == 2)
            EndPrimitive();
        else if (nVerticesFound != 0)  // Debugging purposes (should not happen)
        {
            // Create a long line into below of the object.
            gl_Position = u_projection * u_view * u_model * vec4(0.f, 0.f, u_deepestLevel -10.f, 1.f);
            interpolatedColor = vec4(0.f, 0.f, 0.f, 1.f);
            EmitVertex();
            EndPrimitive();
        }
    }
}

#shader fragment
#version 330 core

in vec4 interpolatedColor;
out vec4 fragColor;

void main()
{
    fragColor = interpolatedColor;
}
