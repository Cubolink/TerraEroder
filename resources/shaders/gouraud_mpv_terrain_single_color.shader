#shader vertex
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec4 vertexColor;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

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

uniform float u_randomNumber;
uniform int u_applyNoise;

uniform vec3 color;


float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 changeVertexPosition(vec3 originalPosition, float minPossibleValue, float valueRange, float extraNum)
{
    //float z = valueRange*(noise1(originalPosition.z)+1) + valueRange*(noise1(extraNum)+1);
    float z = rand(vec2(originalPosition.x, originalPosition.y))+1;
    z += rand(vec2(originalPosition.z, extraNum))+1;
    //z = (z+4)/8;  // [0, valueRange]
    z = valueRange*z/4 + minPossibleValue;

    return vec3(originalPosition.x, originalPosition.y, z);
}

void main()
{
    vec3 noisyPosition = position;
    if (u_applyNoise != 0)
        noisyPosition = changeVertexPosition(position, u_deepestLevel, u_levelRange, u_randomNumber);

    vec3 vertexPos = vec3(u_model * vec4(noisyPosition, 1.0));
    gl_Position = u_projection * u_view * vec4(vertexPos, 1.0);

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
    vec3 result = (ambient + ((diffuse + specular) / attenuation)) * color;
    vertexColor = vec4(result, 1.0);
}

#shader fragment
#version 330 core

in vec4 vertexColor;
out vec4 fragColor;

void main()
{
    fragColor = vertexColor;
}