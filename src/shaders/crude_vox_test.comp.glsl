#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// credit:
// multi-level trace algorithms adapted from amanatides/woo and DapperCore on shadertoy

// problems with the shadows that i need to fix otherwise all is fine

//#include "noise3D.glsl"
#include "shared.inl"

#define BRICK_SIZE 8

uvec3 global_size = uvec3(
    brickmap.data[4],
    brickmap.data[5],
    brickmap.data[6]
);

uint modify_voxel(ivec3 voxel_pos, bool state) {
    uint voxel = 0u;
    uint indices_start = 7u;

    uvec3 brick_n = (global_size + BRICK_SIZE - 1) / BRICK_SIZE;
    uint total_bricks = brick_n.x * brick_n.y * brick_n.z;

    ivec3 b = ivec3(floor(vec3(voxel_pos) / float(BRICK_SIZE)));
    uvec3 brick_i = uvec3(b);
    uvec3 in_brick_i = uvec3(voxel_pos - b * BRICK_SIZE);

    //uvec3 brick_i = (uvec3(voxel_pos) + BRICK_SIZE - 1) / BRICK_SIZE;
    uint brick_index =  brick_i.x + brick_i.y * brick_n.x + brick_i.z * brick_n.x * brick_n.y;

    //uvec3 in_brick_i = uvec3(voxel_pos) % BRICK_SIZE;
    uint in_brick_index = in_brick_i.x + in_brick_i.y * BRICK_SIZE + in_brick_i.z * BRICK_SIZE * BRICK_SIZE;

    uint num_indices = total_bricks;
    uint indices_offset = indices_start + brick_index;

    uint bricks_start = indices_start + num_indices;
    uint buffer_length = glob.brickmap_data_size;

    if (indices_offset >= bricks_start || indices_offset >= buffer_length) {
        return voxel;
    }

    uint indices = brickmap.data[indices_offset];
    uint is_loaded = (indices >> 31) & 1u;
    uint indices_value = indices & 0x7FFFFFFFu;

    if (state == true && is_loaded == 0u) { 
        //uint new_heap_index = new_heap_indices[brick_index];
        brickmap.data[indices_offset] = 0x80000000u | indices_value;
    }

    if (is_loaded == 1u) {
        uint brick_offset = bricks_start + indices_value * BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
        uint voxel_index = brick_offset + in_brick_index;
        if (voxel_index >= buffer_length) {
            return 0u;
        }
        if (state == false) {
            //deref(push.voxel_model).data[indices_offset] = indices_value;
            //deref(push.att.voxel_model).data[indices_offset] = 0x00000000u | indices_value;
            brickmap.data[voxel_index] = 0x00000000u;
        } else if (state == true) {
            brickmap.data[voxel_index] = brick_offset | 0xFF000000u;
        }
        return voxel_index;
    }
    return 0u;
}

uint sample_brick(ivec3 voxel_pos) {
    uint indices_start = 7u;
    uvec3 brick_n = (global_size + BRICK_SIZE - 1) / BRICK_SIZE;
    uint total_bricks = brick_n.x * brick_n.y * brick_n.z;

    ivec3 b = ivec3(floor(vec3(voxel_pos) / float(BRICK_SIZE)));
    uvec3 brick_i = uvec3(b);

    //uvec3 brick_i = (uvec3(voxel_pos) + BRICK_SIZE - 1) / BRICK_SIZE;
    uint brick_index =  brick_i.x + brick_i.y * brick_n.x + brick_i.z * brick_n.x * brick_n.y;

    uint num_indices = total_bricks;
    uint indices_offset = indices_start + brick_index;

    uint bricks_start = indices_start + num_indices;
    uint buffer_length = glob.brickmap_data_size;

    if (indices_offset >= bricks_start || indices_offset >= buffer_length) {
        return 0;
    }

    uint indices = brickmap.data[indices_offset];
    uint is_loaded = (indices >> 31) & 1u;
    uint indices_value = indices & 0x7FFFFFFFu;

    if (is_loaded == 1u) {
        return brick_index;
    }

    return 0;
}

uint sample_brick_voxels(ivec3 voxel_pos, uint brick_index) {
    uint voxel = 0u;
    uint indices_start = 7u;

    uvec3 brick_n = (global_size + BRICK_SIZE - 1) / BRICK_SIZE;
    uint total_bricks = brick_n.x * brick_n.y * brick_n.z;

    uint brick_index_c = brick_index;
    if (brick_index == 0u) {
        ivec3 b = ivec3(floor(vec3(voxel_pos) / float(BRICK_SIZE)));
        uvec3 brick_i = uvec3(b);
        //uvec3 brick_i = (uvec3(voxel_pos) + BRICK_SIZE - 1) / BRICK_SIZE;
        brick_index_c =  brick_i.x + brick_i.y * brick_n.x + brick_i.z * brick_n.x * brick_n.y;
    }

    ivec3 b = ivec3(floor(vec3(voxel_pos) / float(BRICK_SIZE)));
    uvec3 brick_i = uvec3(b);
    uvec3 in_brick_i = uvec3(voxel_pos - b * BRICK_SIZE);
    
    //uvec3 in_brick_i = uvec3(voxel_pos) % BRICK_SIZE;
    uint in_brick_index = in_brick_i.x + in_brick_i.y * BRICK_SIZE + in_brick_i.z * BRICK_SIZE * BRICK_SIZE;

    uint num_indices = total_bricks;
    uint indices_offset = indices_start + brick_index_c;

    uint bricks_start = indices_start + num_indices;
    uint buffer_length = glob.brickmap_data_size;

    if (indices_offset >= bricks_start || indices_offset >= buffer_length) {
        return voxel;
    }

    uint indices = brickmap.data[indices_offset];
    uint is_loaded = (indices >> 31) & 1u;
    uint indices_value = indices & 0x7FFFFFFFu;

    if (is_loaded == 1u) {
        uint brick_offset = bricks_start + indices_value * BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
        uint voxel_index = brick_offset + in_brick_index;

        if (voxel_index >= buffer_length) {
            return voxel;
        }
        voxel = brickmap.data[voxel_index];
    }
    return voxel;
}

hit_info get_voxels(ivec3 p, uint brick_i) {
    p = ivec3(p.x,p.z,p.y);
    uint voxel_colour_packed = sample_brick_voxels(p,brick_i);
    vec4 voxel_colour = unpack_color(voxel_colour_packed);

    if (voxel_colour.a == 0.0) {
        return hit_info(false, vec4(0.0), 0.);
    } else {
        return hit_info(true, voxel_colour, 0.);
    }
    return hit_info(false, vec4(0.0), 0.);
}

uint get_brick(ivec3 p) {
    uint brick_i = 0;
    p = ivec3(p.x,p.z,p.y)*BRICK_SIZE;

    if (p.x <= global_size.x+1 && p.y <= global_size.y+1 && p.z <= global_size.z+1) {
        
        brick_i = sample_brick(p);
    }
    return brick_i;
}

vec3 renderSky(vec3 ro, vec3 rd ) {
    // background sky   
    vec3 col = vec3(0.1922, 0.3765, 0.6118)/0.85 - rd.y*vec3(0.3,0.36,0.4);  
    //vec3 col = vec3(0.0, 0.0, 0.0)/0.85 - rd.y*vec3(0.1,0.0,0.0);  

    // clouds
    float t = (5000.0-ro.y)/rd.y;
    if( rd.y > 0.04 )
    {
        //float cl = snoise(vec3(ro+t*rd)*0.0001);
        float dl = smoothstep(-0.2,0.8,0.);
        col = mix( col, vec3(1.0), 0.3*dl );
    }
    
	return col;
}

shadow_info trace_brick_shadow(vec3 brick_pos, uint brick_i, vec3 rayDir, vec3 iMask, out bool firstVoxel) {
    brick_pos = clamp(brick_pos, vec3(0.001), vec3(BRICK_SIZE-0.001));
    vec3 map_pos = floor(brick_pos);
    ivec3 raySign = ivec3(sign(rayDir));
    vec3 deltaDist = 1.0/rayDir;
    vec3 sideDist = ((map_pos - vec3(brick_pos)) + 0.5 + raySign * 0.5) * deltaDist;
    ivec3 mask = ivec3(iMask);

    while (
        map_pos.x < BRICK_SIZE && map_pos.x >= 0 && 
        map_pos.y < BRICK_SIZE && map_pos.y >= 0 && 
        map_pos.z < BRICK_SIZE && map_pos.z >= 0
    ) {
        //if (map_pos == brick_pos) {break;}

        hit_info h = get_voxels(ivec3(map_pos),brick_i);
        ivec3 mini = ivec3(((map_pos-brick_pos) + 0.5 - 0.5*vec3(raySign))*deltaDist);
        float d = int(floor(max(mini.x, max(mini.y, mini.z))));
        if (h.hit) {
            if (!firstVoxel) {
                return shadow_info(0.4,0.0);
            } else {
                firstVoxel = false;
            }
        }
        mask = ivec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));
        map_pos += mask * raySign;
        sideDist += mask * raySign * deltaDist;
    }
    return shadow_info(0.0,0.0);
}


// simple hash and lighting adapted from parts on shadertoy
// needs to be fully replaced, but works for a demo
float hash(vec3 p) {
    return fract(sin(dot(p ,vec3(12.9898,78.233, 37.719))) * 43758.5453);
}
vec3 light_dir = normalize(vec3(-0.1, 0.2, -0.2));
vec3 light_color = vec3(1.0);
vec3 lighting(vec3 norm, vec3 rd, vec3 col, vec3 pos) {
    vec3 N = normalize(norm);
    vec3 L = normalize(light_dir);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = col * NdotL;

    vec3 ambient = col * 0.8;

    vec3 viewDir = normalize(-rd);
    vec3 reflectDir = reflect(-L, N);
    float baseSpec = max(dot(viewDir, reflectDir), 0.0);

    float ao = 0.5 + 0.5 * N.y;
    ao *= 0.8 + 0.3 * hash(pos * 3.0); 
    
    float noise = hash(pos + norm * 10.0);
    float specular = pow(baseSpec, 16.0) * mix(0.2, 1.0, noise); 

    vec3 color = (diffuse + ambient) + specular * light_color;

    return color;
}

hit_info trace_brick(vec3 brick_pos, uint brick_i, vec3 rayDir, vec3 mask) {
    vec3 brick_pos_c = clamp(brick_pos, vec3(0.0001), vec3(BRICK_SIZE-0.0001));
    ivec3 map_pos = ivec3(floor(brick_pos_c));
    vec3 ray_sign = sign(rayDir);
    vec3 delta_d = 1.0/normalize(rayDir);
    vec3 sideDist = ((map_pos - brick_pos_c) + 0.5 + ray_sign * 0.5) * delta_d;

    while (
        map_pos.x < BRICK_SIZE && map_pos.x >= 0 && 
        map_pos.y < BRICK_SIZE && map_pos.y >= 0 && 
        map_pos.z < BRICK_SIZE && map_pos.z >= 0
    ) {
        if (map_pos == brick_pos_c) {
            break;
        }

        hit_info voxel = get_voxels(map_pos,brick_i);

        vec3 mini = ((map_pos-brick_pos_c) + 0.5 - 0.5*vec3(ray_sign))*delta_d;
        int d = int(floor(max(mini.x, max(mini.y, mini.z))));

        if (voxel.hit) {
            vec4 lighting_result = vec4(lighting(mask, rayDir, voxel.col.rgb, map_pos),1.);
    
            return hit_info(true,lighting_result,d);
        }
        mask = vec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));
        map_pos += ivec3(mask * ray_sign);
        sideDist += mask * ray_sign * delta_d;
    }
    
    return hit_info(false,vec4(0.),0.);
}

shadow_info trace_brickmap_shadows(vec3 ray_origin, vec3 ray_dir) {   
    vec3 ray_pos = ray_origin/BRICK_SIZE;
    vec3 map_pos = floor(ray_pos);
    vec3 ray_sign = sign(ray_dir);
    vec3 deltaDist = 1/ray_dir;
    vec3 sideDist = ((map_pos-ray_pos) + 0.5 + 0.5*ray_sign) * deltaDist;
    vec3 norm = vec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));

    bool firstVoxel = true;
    for (int i = 0; i < 150; i++) {
        vec3 border_min = ivec3(0);
        vec3 border_max = border_min + (ivec3(global_size.x,global_size.z,global_size.y)-7)/8;
        /*if (!(map_pos.x >= border_min.x && map_pos.x < border_max.x &&
            map_pos.y >= border_min.y && map_pos.y < border_max.y &&
            map_pos.z >= border_min.z && map_pos.z < border_max.z)) {
                break;
        }*/

        uint brick_index = get_brick(ivec3(map_pos));

        if (brick_index == 0) {
            break;
        }

        if (brick_index > 0) {
            vec3 mini = ((map_pos-ray_pos) + 0.5 - 0.5*ray_sign)*deltaDist;
            float d = max (mini.x, max (mini.y, mini.z));
            vec3 intersect = ray_pos + ray_dir*d;
            vec3 uv3d = intersect - map_pos;

            if (map_pos == floor(ray_pos))
                uv3d = ray_pos - map_pos;

            shadow_info hit = trace_brick_shadow(uv3d*BRICK_SIZE, brick_index, ray_dir, norm, firstVoxel);

            if (hit.shadow_factor > 0.0) {
                return shadow_info(0.2,hit.d);
            }
        }
        norm = vec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));
        map_pos += norm * ray_sign;
        sideDist += norm * ray_sign * deltaDist;
    }
    
    return shadow_info(1.0,0.0);
}

hit_info trace_brickmap(ray_data ray) { 
    vec3 ray_pos = ray.origin/BRICK_SIZE;
    ray.dir = normalize(ray.dir);

    vec3 map_pos = floor(ray_pos);
    
    vec3 ray_sign = sign(ray.dir);
    vec3 deltaDist = 1.0 / ray.dir;

    vec3 sideDist = ((map_pos-ray_pos) + 0.5 + ray_sign * 0.5) * deltaDist;

    vec3 norm = vec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));
    bool firstVoxel = true;
    for (int i = 0; i < 100; i++) {
        vec3 border_min = ivec3(0);
        vec3 border_max = border_min + (ivec3(global_size.x,global_size.z,global_size.y)-7)/8;
        /*if (!(map_pos.x >= border_min.x && map_pos.x < border_max.x &&
            map_pos.y >= border_min.y && map_pos.y < border_max.y &&
            map_pos.z >= border_min.z && map_pos.z < border_max.z)) {
                break;
        }*/
        uint brick_index = get_brick(ivec3(map_pos));

        if (brick_index == 0) {
            //break;
        }

        vec3 mini = ((map_pos-ray_pos) + 0.5 - 0.5*ray_sign)*deltaDist;
        float d = max(mini.x, max(mini.y, mini.z));
        vec3 intersect = ray_pos + ray.dir*d;
        vec3 uv3d = intersect - map_pos;
        
        if (map_pos == floor(ray_pos))
            uv3d = ray_pos - map_pos;
        
        hit_info hit = trace_brick(uv3d*BRICK_SIZE, brick_index, ray.dir, norm);


        //shadow_info brick_shadow_hit = trace_brick_shadow(uv3d*BRICK_SIZE, brick_i, ray.dir, norm, firstVoxel);
        if (hit.col.a > 0.0) {
            vec3 shadow_intersect = ray.origin + ray.dir * (d*BRICK_SIZE+hit.d);
            shadow_intersect += light_dir * 0.001f;
            //shadow_intersect = 0.5*(floor(shadow_intersect)+ceil(shadow_intersect));
            //hit_info shadow = trace_brickmap_shadows(shadow_intersect,light_dir);
            shadow_info shadow = trace_brickmap_shadows(shadow_intersect, light_dir);

            return hit_info(true,hit.col*shadow.shadow_factor,0.);
        }
        
        norm = vec3(lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy)));
        map_pos += norm * ray_sign;
        sideDist += norm * ray_sign * deltaDist;
    }
    
    return hit_info(false,vec4(0.0),0.);
}

bool cast_to_voxel(in ray_data ray, out ivec3 hit_pos, out vec3 hit_pos_norm) {
    ivec3 border_min, border_max;
    const int maxTrace = 100;

    //by amanatides/woo http://www.cse.yorku.ca/~amana/research/grid.pdf
    ray.origin = ray.origin;
    vec3 tMax;
    vec3 fr = fract(ray.origin);
    vec3 step = vec3(sign(ray.dir));
    vec3 tDelta = step / ray.dir;
    vec3 pos = floor(ray.origin);
    vec3 norm = vec3(0.);

    tMax.x = tDelta.x * ((ray.dir.x>0.0) ? (1.0 - fr.x) : fr.x);
    tMax.y = tDelta.y * ((ray.dir.y>0.0) ? (1.0 - fr.y) : fr.y);
    tMax.z = tDelta.z * ((ray.dir.z>0.0) ? (1.0 - fr.z) : fr.z);

    for (int i = 0; i < maxTrace; i++) {
        hit_info h = get_voxels(ivec3(pos),0u);
        if (h.hit) {
            hit_pos = ivec3(pos);
            hit_pos_norm = norm;
            return true;
        }

        norm = vec3(lessThanEqual(tMax.xyz, min(tMax.yzx, tMax.zxy)));
        tMax += norm * tDelta;
        pos += norm * step;
    }
 	return false;
}

struct collision_info {
    bool is_colliding;
    bool is_on_ground;
    bool is_ceiling_low;
    vec3 offset;
};

bool check_corner(in ivec3 origin, in ivec3 corner, out vec3 offset) {
    if(get_voxels(ivec3(corner),0).hit) {
        offset = normalize(vec3(origin-corner))*0.001;
        offset.y += 0.001;
        return true;
    }

    return false;
}
/*
collision_info aabb_intersect(vec3 origin_f) {
    collision_info info;
    info.is_on_ground = false;
    info.is_colliding = false;

    ivec3 origin = ivec3(floor(origin_f));

    vec3 offset = vec3(0.);
    ivec3 half_ext = ivec3(8,24,8);

    ivec3 ground = ivec3(origin.x, origin.y-half_ext.y, origin.z);

    if(check_corner(origin,ground,offset)) {
        info.is_on_ground = true;
    }

    ivec3 corner_b1 = ivec3(origin.x-half_ext.x, origin.y-half_ext.y, origin.z-half_ext.z);
    ivec3 corner_b2 = ivec3(origin.x-half_ext.x, origin.y-half_ext.y, origin.z+half_ext.z);
    ivec3 corner_b3 = ivec3(origin.x+half_ext.x, origin.y-half_ext.y, origin.z+half_ext.z);
    ivec3 corner_b4 = ivec3(origin.x+half_ext.x, origin.y-half_ext.y, origin.z-half_ext.z);
    ivec3 corner_t1 = ivec3(origin.x-half_ext.x, origin.y+half_ext.y, origin.z-half_ext.z);
    ivec3 corner_t2 = ivec3(origin.x-half_ext.x, origin.y+half_ext.y, origin.z+half_ext.z);
    ivec3 corner_t3 = ivec3(origin.x+half_ext.x, origin.y+half_ext.y, origin.z+half_ext.z);
    ivec3 corner_t4 = ivec3(origin.x+half_ext.x, origin.y+half_ext.y, origin.z-half_ext.z);

    if(check_corner(origin,corner_b1,offset) 
    || check_corner(origin,corner_b2,offset) 
    || check_corner(origin,corner_b3,offset)
    || check_corner(origin,corner_b4,offset)) {
        info.is_colliding = true;
        info.is_on_ground = true;
        //info.offset += offset;
        
    }

    info.offset = offset;
    
    return info;
}*/

void main() {
    ivec2 img_coord = ivec2(gl_GlobalInvocationID.x,gl_GlobalInvocationID.y);
    vec2 uv = (vec2(img_coord) / imageSize(img).xy);
    
    mat4 view = inverse(glob.view);
    mat4 proj = glob.proj;
    mat4 clip_to_world = transpose(inverse(view*proj));
    float proj_fov = -(3.141/180)*4*atan(proj[1][1]);
    vec4 clip_space_coord = vec4((uv.x - 0.5), - (uv.y - 0.5), 0.0, proj_fov);
    uint value = brickmap.data[0];
    //vec3 ray_origin = view[3].xyz;
    vec3 ray_origin = vec3(view[3][0],view[3][1],view[3][2]);
    vec3 ray_dir = (clip_to_world*clip_space_coord).xyz;
    ray_data ray = ray_data(ray_origin,ray_dir);

    // edit
    vec4 front_vector_coord = vec4(0, 0, -1, proj_fov);
    vec3 front_vector = (clip_to_world*front_vector_coord).xyz;
    ray_data front_ray = ray_data(ray_origin,front_vector);

    ivec3 hit_pos = ivec3(0.);
    vec3 hit_pos_norm = vec3(0.);

    cast_to_voxel(front_ray, hit_pos, hit_pos_norm);
    
    ivec3 hit_pos_c = ivec3(hit_pos.x,hit_pos.z,hit_pos.y);

    bool place_state = (glob.player_flags & (1u << 1)) != 0u;
    bool break_state = (glob.player_flags & (1u << 0)) != 0u;
    
    //bool is_colliding = false;
    //bool is_on_ground = false;
    //bool is_ceiling_low = false;
    //vec3 collision_offset = vec3(0.);

    //collision_info info = aabb_intersect(ray_origin);
    //collision_offset = info.offset;
    //gpu_out.collision_offset = collision_offset;
    
    if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
        //bool is_colliding = info.is_colliding;
        //bool is_on_ground = info.is_on_ground;
        //bool is_ceiling_low = info.is_ceiling_low;

        //gpu_out.collision_flags = 
        //    (uint(is_colliding) << 0) |
        //    (uint(is_on_ground) << 1) |
        //    (uint(is_ceiling_low) << 2);
        
        ivec3 voxel_pos = ivec3(0);
        int size = 3;
        if (break_state || place_state) {
            for (int x = -size; x <= size; x++) {
                for (int y = -size; y <= size; y++) {
                    for (int z = -size; z <= size; z++) {
                        voxel_pos = ivec3(x,y,z);
                            if (break_state) {
                                modify_voxel(hit_pos_c+voxel_pos, false);
                            } else if (place_state) {
                                modify_voxel(hit_pos_c+voxel_pos, true);
                            }
                    }
                }
            }
        }
    }
    
    vec4 color = vec4(0.);
    hit_info h = trace_brickmap(ray);
    color = h.hit ? h.col : vec4(renderSky(ray.origin,ray.dir),0.);
    //color = vec4(renderSky(ray.origin,ray.dir),0.);
    color.rgb = ACESFilm(color.rgb);
    color.rgb = pow( clamp(color.rgb*1.1-0.02,0.0,1.0), vec3(1.2) ); //gamma
    color.rgb = color.rgb*color.rgb*(3.0-2.0*color.rgb); //contrast

    imageStore(img,  img_coord, color);
}