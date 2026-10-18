#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "NeighbourList.hpp"
#include "Renderer.hpp"
#include "System.hpp"
#include "Thermostat.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string_view>
#include <vector>

// This file contains the main function for the molecular dynamics simulation program. 
// It sets up the simulation parameters, initialises the system, and runs the simulation loop. 
// The program can be run with various command line options to customise the simulation.

namespace {

// Reduced Lennard-Jones units, same state point as md_sim
constexpr double density = 0.8442;
constexpr double initialTemperature = 0.728;
constexpr double cutOff = 2.5;
constexpr double skin = 0.2;
constexpr double dt = 0.005;
constexpr int rescaleEvery = 10;

struct App {
    int cellsPerSide;
    double latticeConstant;
    double boxLength;

    System system;
    LennardJonesForceField forceField {1.0, 1.0, cutOff};
    VelocityVerletIntegrator integrator {dt};
    NeighbourList neighbourList {cutOff, skin};
    double potential {0.0};
    long long step {0};

    Camera camera;
    bool paused {false};
    bool thermostat {true};
    double targetTemperature {initialTemperature};
    int stepsPerFrame {5};

    bool dragging {false};
    double lastX {0.0};
    double lastY {0.0};

    explicit App(int cells)
        : cellsPerSide(cells),
          latticeConstant(std::cbrt(4.0 / density)),
          boxLength(cells * latticeConstant),
          system(createFCCLattice(cells, latticeConstant), boxLength) {
        reset();
        const float centre = static_cast<float>(0.5 * boxLength);
        camera.target = {centre, centre, centre};
        camera.distance = static_cast<float>(2.2 * boxLength);
    }

    // Back to a perfect lattice at the starting temperature, so the melt can be watched again
    void reset() {
        system = System(createFCCLattice(cellsPerSide, latticeConstant), boxLength);
        initializeVelocities(system, initialTemperature, 42);
        targetTemperature = initialTemperature;
        neighbourList.build(system);
        potential = forceField.computeForces(system, neighbourList);
        step = 0;
    }

    void advance() {
        for (int s = 0; s < stepsPerFrame; ++s) {
            potential = integrator.step(system, forceField, neighbourList);
            if (thermostat && ++step % rescaleEvery == 0) {
                rescaleVelocities(system, targetTemperature);
            }
        }
    }
};

// Save the frame just drawn (before the buffer swap) as a binary PPM image
bool saveScreenshot(const char* path, int width, int height) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(3 * width * height));
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::FILE* file = std::fopen(path, "wb");
    if (!file) {
        return false;
    }
    std::fprintf(file, "P6\n%d %d\n255\n", width, height);
    // OpenGL rows start at the bottom, image rows at the top
    for (int row = height - 1; row >= 0; --row) {
        std::fwrite(pixels.data() + static_cast<std::size_t>(3 * width * row), 1, static_cast<std::size_t>(3 * width), file);
    }
    std::fclose(file);
    return true;
}

App& app(GLFWwindow* window) {
    return *static_cast<App*>(glfwGetWindowUserPointer(window));
}

void onKey(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }
    App& a = app(window);
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_SPACE: a.paused = !a.paused; break;
        case GLFW_KEY_UP: a.stepsPerFrame = std::min(a.stepsPerFrame + 1, 100); break;
        case GLFW_KEY_DOWN: a.stepsPerFrame = std::max(a.stepsPerFrame - 1, 1); break;
        case GLFW_KEY_H: a.targetTemperature *= 1.1; a.thermostat = true; break;
        case GLFW_KEY_C: a.targetTemperature /= 1.1; a.thermostat = true; break;
        case GLFW_KEY_T: a.thermostat = !a.thermostat; break;
        case GLFW_KEY_R: a.reset(); break;
        default: break;
    }
}

void onMouseButton(GLFWwindow* window, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        App& a = app(window);
        a.dragging = action == GLFW_PRESS;
        glfwGetCursorPos(window, &a.lastX, &a.lastY);
    }
}

void onCursor(GLFWwindow* window, double x, double y) {
    App& a = app(window);
    if (a.dragging) {
        a.camera.yaw -= 0.005f * static_cast<float>(x - a.lastX);
        a.camera.pitch = std::clamp(a.camera.pitch + 0.005f * static_cast<float>(y - a.lastY), -1.5f, 1.5f);
    }
    a.lastX = x;
    a.lastY = y;
}

void onScroll(GLFWwindow* window, double, double yOffset) {
    App& a = app(window);
    const float minDistance = 0.2f * static_cast<float>(a.boxLength);
    const float maxDistance = 10.0f * static_cast<float>(a.boxLength);
    a.camera.distance = std::clamp(a.camera.distance * std::pow(0.9f, static_cast<float>(yOffset)), minDistance, maxDistance);
}

}

int main(int argc, char** argv) {
    int cellsPerSide = 8;
    const char* screenshotPath = nullptr;
    int screenshotFrames = 300;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--screenshot" && i + 1 < argc) {
            screenshotPath = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            screenshotFrames = std::atoi(argv[++i]);
        } else {
            cellsPerSide = std::atoi(argv[i]);
        }
    }
    if (cellsPerSide < 2 || screenshotFrames < 1) {
        std::fprintf(stderr, "Usage: md_vis [cellsPerSide >= 2] [--screenshot PATH.ppm] [--frames N]\n");
        return 1;
    }

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialise GLFW\n");
        return 1;
    }

    // OpenGL 4.1 core profile: the newest version macOS supports
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 800, "md_vis", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create an OpenGL 4.1 window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    int status = 0;
    try {
        App a(cellsPerSide);
        Renderer renderer;

        glfwSetWindowUserPointer(window, &a);
        glfwSetKeyCallback(window, onKey);
        glfwSetMouseButtonCallback(window, onMouseButton);
        glfwSetCursorPosCallback(window, onCursor);
        glfwSetScrollCallback(window, onScroll);

        std::printf("N = %zu. Drag to orbit, scroll to zoom, Space pause, Up/Down steps per frame,\n"
                    "H heat / C cool, T thermostat on/off, R reset, Esc quit\n", a.system.getParticles().size());

        // Speed colours are fixed to the starting temperature, so heating visibly brightens the particles
        const double referenceSpeed = std::sqrt(3.0 * initialTemperature);

        double lastTitle = glfwGetTime();
        int frames = 0;
        int totalFrames = 0;

        while (!glfwWindowShouldClose(window)) {
            if (!a.paused) {
                a.advance();
            }

            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            renderer.draw(a.system, a.camera, width, height, referenceSpeed);

            if (screenshotPath && ++totalFrames == screenshotFrames) {
                if (!saveScreenshot(screenshotPath, width, height)) {
                    std::fprintf(stderr, "Could not write %s: %s\n", screenshotPath, std::strerror(errno));
                    status = 1;
                }
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();

            // Stats in the window title, updated a few times a second
            ++frames;
            const double now = glfwGetTime();
            if (now - lastTitle > 0.25) {
                const double n = static_cast<double>(a.system.getParticles().size());
                char title[256];
                std::snprintf(title, sizeof(title),
                    "md_vis | N = %zu | T = %.3f (target %.3f, thermostat %s) | E/N = %.3f | %d steps/frame | %.0f fps%s",
                    a.system.getParticles().size(), a.system.temperature(), a.targetTemperature,
                    a.thermostat ? "on" : "off", (a.system.kineticEnergy() + a.potential) / n, a.stepsPerFrame,
                    frames / (now - lastTitle), a.paused ? " | paused" : "");
                glfwSetWindowTitle(window, title);
                lastTitle = now;
                frames = 0;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        status = 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return status;
}
