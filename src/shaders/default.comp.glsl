#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "shared.inl"

void main() {
    ivec2 img_coord = ivec2(gl_GlobalInvocationID.x,gl_GlobalInvocationID.y);
    vec2 uv = vec2(img_coord-0.5) / imageSize(img).xy;

    mat4 proj = glob.proj;
    uint value = brickmap.data[0];
    vec4 pixel = vec4(0.);

    imageStore(img,  img_coord, pixel);
}