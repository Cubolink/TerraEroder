//
// Created by major on 17-12-2021.
//

#ifndef TERRAINGEN_CONTROLER_H
#define TERRAINGEN_CONTROLER_H

#include "camera.h"


/**
 * Class to control camera properties
 */
class CameraController
{
private:
    Camera *camera;

public:
    /// variables to determine if the camera should move in what directions.
    float m_left = 0, m_right = 0,
    m_forth = 0, m_back = 0,
    m_up = 0, m_down = 0,
    rot_up = 0, rot_down = 0,
    rot_left = 0, rot_right = 0;

    /**
     * Creates an instance of a Camera Controller
     */
    CameraController()
    : camera(nullptr)
    {

    }

    /**
     * Sets a camera to control
     * @param cam
     */
    void setCamera(Camera *cam)
    {
        camera = cam;
    }

    /**
     * Using the public variables, updates the camera properties
     */
    void updateCameraProperties()
    {
        if (camera == nullptr)
            return;
        float forward_strength = (m_forth - m_back);
        camera->dx_dt = forward_strength * (cosf(camera->phi));  // * (sinf(camera->theta));
        camera->dy_dt = forward_strength * (sinf(camera->phi));  // * (sinf(camera->theta));
//        camera->dz_dt = forward_speed * (cosf(camera->theta));

        float side_strength = (m_right - m_left);
        camera->dx_dt += abs(side_strength) * (cosf(camera->phi - (float) M_PI_2 * side_strength));
        camera->dy_dt += abs(side_strength) * (sinf(camera->phi - (float) M_PI_2 * side_strength));

        // normalize (dx_dt. dy_dt)
        double module = camera->dx_dt * camera->dx_dt + camera->dy_dt * camera->dy_dt;
        double normalization_factor = module != 0 ? camera->max_dEye_dt / module: 0;
        camera->dx_dt = (float) normalization_factor * camera->dx_dt;
        camera->dy_dt = (float) normalization_factor * camera->dy_dt;

        camera->dz_dt = camera->max_dCenter_dt * (m_up - m_down);

        camera->dPhi_dt = camera->max_dCenter_dt * (rot_left - rot_right);
        camera->dTheta_dt = camera->max_dCenter_dt * (rot_down - rot_up);
    }



};

#endif //TERRAINGEN_CONTROLER_H
