//
// Created by major on 16-12-2021.
//

#ifndef TERRAINGEN_CAMERA_H
#define TERRAINGEN_CAMERA_H

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


/**
 * A first person camera. It uses spherical coordinates to handle its logic.
 */
class Camera
{
private:
    glm::vec3 eye;                             /// position of the camera
    float dx_dt, dy_dt, dz_dt, max_dEye_dt;    /// speed of the camera's position
    float phi, theta;                          /// Sphere coordinates, to determine where is the center where it's looking at
    float dPhi_dt, dTheta_dt, max_dCenter_dt;  /// speed of the camera's angles

public:
    /**
     * Creates an instance of the camera.
     */
    Camera();

    /**
     * Gets the position where the camera is placed
     */
    glm::vec3 getEyeVec3();

    /**
     * Gets the position to where the camera is pointing
     */
    glm::vec3 getCenterVec3() const;

    /**
     * Gets an orientation vector for the camera
     */
    glm::vec3 getUpVec3();

    /**
     * Gets the current view matrix of the ModelViewProjection.
     */
    glm::mat4 getViewMatrix();

    /**
     * Recalculates eye and center position using the speeds and the dt parameter
     * @param dt delta time
     */
    void updateCoords(float dt);

    /**
     * @return camera's phi (spherical coords), phi-coordinate of the center look at
     */
    inline float getPhi() const {return phi;}

    /**
     * @return camera's theta (spherical coords), theta-coordinate of the center look at
     */
    inline float getTheta() const {return theta;}

    /**
     * @return x-coordinate of the camera's eye
     */
    inline float getCX() const {return eye.x;}

    /**
     * @return y-coordinate of the camera's eye
     */
    inline float getCY() const {return eye.y;}

    /**
     * @return z-coordinate of the camera's eye
     */
    inline float getCZ() const {return eye.z;}

    void setPhi(float value);

    void setTheta(float value);


    /**
     * Sets the eye of the camera to a new position
     * @param x new coordinate
     * @param y new coordinate
     * @param z new coordinate
     */
    void setEye(float x, float y, float z);

    /**
     * Sets a new position to where the camera will point at.
     * @param x new coordinate
     * @param y new coordinate
     * @param z new coordinate
     */
    void setCenter(float x, float y, float z);

    friend class CameraController;
};

#endif //TERRAINGEN_CAMERA_H
