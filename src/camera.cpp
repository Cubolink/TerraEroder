//
// Created by major on 16-12-2021.
//

#include "camera.h"

#include <iostream>
#include <cmath>

Camera::Camera()
: eye(0, 0, 0), phi(0), theta(0),
dx_dt(0), dy_dt(0), dz_dt(0), max_dEye_dt(2),
dPhi_dt(0), dTheta_dt(0), max_dCenter_dt(2)
{

}

glm::vec3 Camera::getEyeVec3() {
    return eye;
}

glm::vec3 Camera::getCenterVec3() const {
    return {std::cos(phi) * std::sin(theta) + eye.x,
            std::sin(phi) * std::sin(theta) + eye.y,
            std::cos(theta) + eye.z};
}

glm::vec3 Camera::getUpVec3() {
    return {.0f, .0f, 1.0f};
}

glm::mat4 Camera::getViewMatrix() {
    return glm::lookAt(getEyeVec3(), getCenterVec3(), getUpVec3());
}

void Camera::setPhi(float value) {
    phi = value;
}

void Camera::setTheta(float value) {
    theta = value;
}

void Camera::setEye(float x, float y, float z) {
    eye.x = x;
    eye.y = y;
    eye.z = z;
}

void Camera::setCenter(float x, float y, float z) {
    x -= eye.x;
    y -= eye.y;
    z -= eye.z;

    // rho = (float) std::sqrt(x*x + y*y + z*z);
    phi = (float) std::atan2(y, x);
    theta = (float) std::atan(std::sqrt(x*x + y*y)/std::sqrt(z*z));
    if (z < 0)
        theta = glm::radians(180.0f) - theta;

}

void Camera::updateCoords(float dt) {
    eye.x += dx_dt * dt;
    eye.y += dy_dt * dt;
    eye.z += dz_dt * dt;
    phi += dPhi_dt * dt;
    theta += dTheta_dt * dt;
    if (phi < 0 || phi > 2 * M_PI) phi = (float) fmod(phi, 2 * M_PI);
    if (theta <= 0) theta = 0.0001;
    else if (theta >= M_PI) theta = M_PI - 0.0001;
}

