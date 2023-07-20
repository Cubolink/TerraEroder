#shader vertex
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float waterHeight;
layout(location = 3) in float sediment;

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

uniform float u_sedimentColorIntensity;

void main()
{
    vec3 vertexPos = vec3(u_model * vec4(position.x, position.y, position.z + waterHeight, 1.0));
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
    vec3 blue = vec3(0.2666f, 0.7333f, 1.f);
    vec3 brown = vec3(0.5255f, 0.3490f, 0.1137f);
    float pond = min(1.f, u_sedimentColorIntensity * sediment/waterHeight);
    vec3 color = pond * brown + (1 - pond) * blue;

    vec3 result = (ambient + ((diffuse + specular) / attenuation)) * color;
    vertexColor = vec4(result, min(0.3f, waterHeight));
}

#shader fragment
#version 330 core

in vec4 vertexColor;
out vec4 fragColor;

void main()
{
    fragColor = vertexColor;
}