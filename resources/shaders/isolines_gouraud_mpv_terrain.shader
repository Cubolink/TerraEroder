#shader vertex
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec4 shadingResult;

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

    shadingResult = vec4(ambient + ((diffuse + specular) / attenuation), 1.f);
}

#shader geometry
#version 330 core

layout(triangles) in;
layout(line_strip, max_vertices=40) out;

in vec4 shadingResult[];

out float isolineZLevel;
out vec4 interpolatedShading;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_deepestLevel;
uniform float u_levelRange;
uniform int u_nIsolines;  // This value can be up to 20. More than 20 will have no effect.
uniform float u_isolines[20];

vec4 interpolate(vec4 A, vec4 B, float lambda)
{
    return A + lambda*(B - A);
}

#define A gl_in[0].gl_Position
#define B gl_in[1].gl_Position
#define C gl_in[2].gl_Position
#define shadingA vec4(shadingResult[0])
#define shadingB vec4(shadingResult[1])
#define shadingC vec4(shadingResult[2])


void main() {

    int nIsolines = min(20, u_nIsolines);  //
    for (int kIsoline = 0; kIsoline < nIsolines; kIsoline++)
    {
        // Find the two points where the triangle intersect the isoline, if any.
        //isolineZLevel = u_deepestLevel + u_levelRange * float(kIsoline) / nIsolines;
        isolineZLevel = u_isolines[kIsoline];
        // Check if AB, BC or CA are parallel to the isolineZLevel. If true, both extrems will form the primitive
        if ((B.z - A.z == 0) && A.z == isolineZLevel)
        {
            gl_Position = u_projection * u_view * u_model * A; interpolatedShading = shadingA;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * B; interpolatedShading = shadingB;
            EmitVertex();
            EndPrimitive();
            continue;
        } else if ((C.z - B.z == 0) && B.z == isolineZLevel)
        {
            gl_Position = u_projection * u_view * u_model * B; interpolatedShading = shadingB;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * C; interpolatedShading = shadingC;
            EmitVertex();
            EndPrimitive();
            continue;
        } else if ((A.z - C.z == 0 && C.z == isolineZLevel))
        {
            gl_Position = u_projection * u_view * u_model * C; interpolatedShading = shadingC;
            EmitVertex();
            gl_Position = u_projection * u_view * u_model * A; interpolatedShading = shadingA;
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
                interpolatedShading = interpolate(shadingA, shadingB, lambda);
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
                interpolatedShading = interpolate(shadingB, shadingC, lambda);
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
                interpolatedShading = interpolate(shadingC, shadingA, lambda);
                EmitVertex();
                nVerticesFound++;
            }
        }
        if (nVerticesFound == 2)
            EndPrimitive();
    }
}

#shader fragment
#version 330 core

in float isolineZLevel;
in vec4 interpolatedShading;
out vec4 fragColor;

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
    float f = fmod(hsv_color.x/60.0, 6) - h_i;

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
    // color
    vec3 color;
    if (isolineZLevel < u_waterLevel)
    {
        color = vec3(0, 0, 1);
    }
    else
    {
        float hue = 240 * (isolineZLevel - u_deepestLevel) / u_levelRange;
        color = hsv_to_rgb(vec3(240.f - hue, 1.f, .5f));
    }
    fragColor = interpolatedShading * vec4(color, 1.f);
}
