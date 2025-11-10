layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 0, binding = 1) uniform GlobalUniforms {
    mat4 view;
    mat4 proj;
    float time;
    uint brickmap_data_size;
    uint player_flags;
    float _padding[1];
} glob;

layout(set = 0, binding = 2) buffer GPUOut {
    vec3 collision_offset;
    uint collision_flags;
    float _padding[3];
} gpu_out;

layout(set = 0, binding = 3) buffer BrickMap {
    uint data[];
} brickmap;

struct hit_info {
    bool hit;
    vec4 col;
    float d;
};

struct shadow_info {
    float shadow_factor;
    float d;
};

struct ray_data {
    vec3 origin;
    vec3 dir;
};

vec4 unpack_color(uint color) {
    float r   = float((color >> 24) & 0xFF) / 255.0;
    float g = float((color >> 16) & 0xFF) / 255.0; 
    float b  = float((color >> 8) & 0xFF) / 255.0; 
    float a = float(color & 0xFF) / 255.0;     
    return vec4(r,g,b,a);
    //return vec4(r,g,b,a);
}

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0., 1.);
}