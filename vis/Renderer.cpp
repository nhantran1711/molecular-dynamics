#include "Renderer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

// Column-major 4x4 matrix, as OpenGL expects
using Mat4 = std::array<float, 16>;

// Create a perspective projection matrix with the given vertical field of view, aspect ratio, and near/far planes
Mat4 perspective(float fovY, float aspect, float zNear, float zFar) {
    const float f = 1.0f / std::tan(0.5f * fovY);
    Mat4 m {};
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zFar + zNear) / (zNear - zFar);
    m[11] = -1.0f;
    m[14] = 2.0f * zFar * zNear / (zNear - zFar);
    return m;
}

// Create a view matrix that looks from the eye position to the target position, with the up vector pointing in the positive y direction
Mat4 lookAt(const std::array<float, 3>& eye, const std::array<float, 3>& target) {
    auto normalise = [](std::array<float, 3> v) {
        float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return std::array<float, 3> {v[0] / len, v[1] / len, v[2] / len};
    };
    auto cross = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
        return std::array<float, 3> {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    auto dot = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    // Compute the forward, right, and up vectors for the camera coordinate system
    const auto forward = normalise({target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]});
    const auto right = normalise(cross(forward, {0.0f, 1.0f, 0.0f}));
    const auto up = cross(right, forward);

    return {
        right[0], up[0], -forward[0], 0.0f, right[1],         up[1],         -forward[1],       0.0f,
        right[2], up[2], -forward[2], 0.0f, -dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0f,
    };
}

// Sequential blue ramp, dark to light: slow particles recede into the dark background, fast ones glow
constexpr std::array<float, 3> slowColour {0x18 / 255.0f, 0x4f / 255.0f, 0x95 / 255.0f}; // #184f95
constexpr std::array<float, 3> fastColour {0xcd / 255.0f, 0xe2 / 255.0f, 0xfb / 255.0f}; // #cde2fb
constexpr std::array<float, 3> boxColour {0x6b / 255.0f, 0x6a / 255.0f, 0x66 / 255.0f};  // #6b6a66
constexpr std::array<float, 3> background {0x1a / 255.0f, 0x1a / 255.0f, 0x19 / 255.0f}; // #1a1a19

constexpr float particleRadius = 0.5f; // sigma / 2, so touching spheres are one sigma apart

const char* particleVertexShader = R"(
#version 410 core
layout(location = 0) in vec3 position;
layout(location = 1) in float speed;

uniform mat4 view;
uniform mat4 projection;
uniform float radius;
uniform float viewportHeight;

out float vSpeed;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    // Projected diameter in pixels, so spheres shrink with distance like real geometry
    gl_PointSize = viewportHeight * projection[1][1] * radius / gl_Position.w;
    vSpeed = speed;
}
)";

const char* particleFragmentShader = R"(
#version 410 core
in float vSpeed;

uniform vec3 slowColour;
uniform vec3 fastColour;

out vec4 fragColour;

void main() {
    // Treat the square point sprite as a sphere seen head-on: discard the corners, light the rest
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    p.y = -p.y;
    float r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }
    vec3 normal = vec3(p, sqrt(1.0 - r2));
    vec3 light = normalize(vec3(0.4, 0.6, 1.0));
    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 32.0);

    vec3 base = mix(slowColour, fastColour, vSpeed);
    fragColour = vec4(base * (0.3 + 0.7 * diffuse) + 0.2 * specular, 1.0);
}
)";

const char* boxVertexShader = R"(
#version 410 core
layout(location = 0) in vec3 position;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
}
)";

const char* boxFragmentShader = R"(
#version 410 core
uniform vec3 colour;
out vec4 fragColour;
void main() {
    fragColour = vec4(colour, 1.0);
}
)";

// Compile a shader of the given type (vertex or fragment) from the provided source code, and return its OpenGL handle
GLuint compileShader(GLenum type, const char* source) {

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compilation errors and throw an exception if any are found
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        glDeleteShader(shader);
        throw std::runtime_error(std::string("Shader compile failed: ") + log);
    }
    return shader;
}

// Link a vertex and fragment shader into a shader program, and return its OpenGL handle
GLuint linkProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    // Link the shaders into a program and check for linking errors
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        glDeleteProgram(program);
        throw std::runtime_error(std::string("Shader link failed: ") + log);
    }
    return program;
}

} // namespace

Renderer::Renderer() {
    particleProgram_ = linkProgram(particleVertexShader, particleFragmentShader);
    boxProgram_ = linkProgram(boxVertexShader, boxFragmentShader);

    // Particles: interleaved x, y, z, speed
    glGenVertexArrays(1, &particleVao_);
    glGenBuffers(1, &particleVbo_);
    glBindVertexArray(particleVao_);
    glBindBuffer(GL_ARRAY_BUFFER, particleVbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    // OpenGL takes the byte offset into the bound buffer as a pointer
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Box: 12 edges as line segments
    glGenVertexArrays(1, &boxVao_);
    glGenBuffers(1, &boxVbo_);
    glBindVertexArray(boxVao_);
    glBindBuffer(GL_ARRAY_BUFFER, boxVbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
}

Renderer::~Renderer() {
    glDeleteBuffers(1, &particleVbo_);
    glDeleteBuffers(1, &boxVbo_);
    glDeleteVertexArrays(1, &particleVao_);
    glDeleteVertexArrays(1, &boxVao_);
    glDeleteProgram(particleProgram_);
    glDeleteProgram(boxProgram_);
}

void Renderer::uploadBox(double boxLength) {
    const float l = static_cast<float>(boxLength);
    std::vector<float> lines;
    // Each edge joins two corners that differ in exactly one coordinate
    for (unsigned a = 0; a < 8; ++a) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            unsigned b = a | (1u << axis);
            if (b == a) {
                continue;
            }
            for (unsigned corner : {a, b}) {
                lines.push_back((corner & 1u) ? l : 0.0f);
                lines.push_back((corner & 2u) ? l : 0.0f);
                lines.push_back((corner & 4u) ? l : 0.0f);
            }
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, boxVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(lines.size() * sizeof(float)), lines.data(), GL_STATIC_DRAW);
    boxLength_ = boxLength;
}

void Renderer::draw(const System& system, const Camera& camera, int framebufferWidth, int framebufferHeight,
                    double referenceSpeed) {
    if (system.getBoxLength() != boxLength_) {
        uploadBox(system.getBoxLength());
    }

    // Pack positions and speeds as floats. Speed maps to [0, 1] with referenceSpeed at the middle
    const std::size_t n = system.size();
    vertices_.resize(4 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto r = system.positionOf(i);
        const auto u = system.velocityOf(i);
        const double v = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        vertices_[4 * i + 0] = static_cast<float>(r[0]);
        vertices_[4 * i + 1] = static_cast<float>(r[1]);
        vertices_[4 * i + 2] = static_cast<float>(r[2]);
        vertices_[4 * i + 3] = static_cast<float>(std::clamp(0.5 * v / referenceSpeed, 0.0, 1.0));
    }

    glBindBuffer(GL_ARRAY_BUFFER, particleVbo_);
    const auto bytes = static_cast<GLsizeiptr>(vertices_.size() * sizeof(float));
    if (n > particleCapacity_) {
        glBufferData(GL_ARRAY_BUFFER, bytes, vertices_.data(), GL_DYNAMIC_DRAW);
        particleCapacity_ = n;
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices_.data());
    }

    // Camera on a sphere around the target
    const std::array<float, 3> eye {
        camera.target[0] + camera.distance * std::cos(camera.pitch) * std::sin(camera.yaw),
        camera.target[1] + camera.distance * std::sin(camera.pitch),
        camera.target[2] + camera.distance * std::cos(camera.pitch) * std::cos(camera.yaw),
    };
    const float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(std::max(framebufferHeight, 1));
    const Mat4 view = lookAt(eye, camera.target);
    const Mat4 projection = perspective(camera.fovY, aspect, 0.1f, 10.0f * camera.distance);

    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glClearColor(background[0], background[1], background[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(particleProgram_);
    glUniformMatrix4fv(glGetUniformLocation(particleProgram_, "view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(particleProgram_, "projection"), 1, GL_FALSE, projection.data());
    glUniform1f(glGetUniformLocation(particleProgram_, "radius"), particleRadius);
    glUniform1f(glGetUniformLocation(particleProgram_, "viewportHeight"), static_cast<float>(framebufferHeight));
    glUniform3fv(glGetUniformLocation(particleProgram_, "slowColour"), 1, slowColour.data());
    glUniform3fv(glGetUniformLocation(particleProgram_, "fastColour"), 1, fastColour.data());
    glBindVertexArray(particleVao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(n));

    glUseProgram(boxProgram_);
    glUniformMatrix4fv(glGetUniformLocation(boxProgram_, "view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(boxProgram_, "projection"), 1, GL_FALSE, projection.data());
    glUniform3fv(glGetUniformLocation(boxProgram_, "colour"), 1, boxColour.data());
    glBindVertexArray(boxVao_);
    glDrawArrays(GL_LINES, 0, 24);

    glBindVertexArray(0);
}
