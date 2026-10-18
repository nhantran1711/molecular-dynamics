#pragma once

#include "System.hpp"
#include <array>
#include <cstddef>
#include <vector>

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

// Orbit camera looking at a target point, controlled by yaw/pitch angles and distance
struct Camera {
    std::array<float, 3> target {0.0f, 0.0f, 0.0f};
    float yaw {0.6f};        // radians around the vertical axis
    float pitch {0.4f};      // radians above the horizontal plane
    float distance {30.0f};
    float fovY {0.8f};       // vertical field of view, radians
};

// Draws particles as sphere-shaded point sprites (one vertex per particle) and the simulation box as lines.
// Needs a current OpenGL 4.1 core context when constructed and when drawing.
class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Draw one frame. Particles are coloured by speed, with referenceSpeed mapped to the middle of the ramp
    void draw(const System& system, const Camera& camera, int framebufferWidth, int framebufferHeight,
        double referenceSpeed);

private:
    void uploadBox(double boxLength);

    GLuint particleProgram_ {0};
    GLuint boxProgram_ {0};
    GLuint particleVao_ {0};
    GLuint particleVbo_ {0};
    GLuint boxVao_ {0};
    GLuint boxVbo_ {0};

    std::size_t particleCapacity_ {0};
    double boxLength_ {0.0};

    // x, y, z, normalised speed per particle, rebuilt every frame
    std::vector<float> vertices_;
};
