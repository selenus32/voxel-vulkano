use dot_vox::*;
use glam::{IVec3, UVec3, Vec3, Mat3};

const BRICK_SIZE: usize = 8;
const CELL_SIZE: usize = 1;
const BRICK_VOL: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;

#[derive(Clone)]
struct Brick {
    data: [u32; BRICK_VOL],
}

use crate::voxels::magicavoxel::*;

pub fn parse_vox_to_brk(data: &dot_vox::DotVoxData) -> Vec<u8> {
    if data.models.is_empty() {
        println!("Magicavoxel models are empty");
        return Vec::new();
    }

    // taken from dot_vox/examples/traverse_graph.rs
    let mut min_extent = IVec3::splat(i32::MAX);
    let mut max_extent = IVec3::splat(i32::MIN);

    iterate_vox_tree(data, |model, translation, orientation| {
        let extent = IVec3::new(
            model.size.x as i32,
            model.size.y as i32,
            model.size.z as i32
        );

        let corners = [
            IVec3::new(0, 0, 0),
            IVec3::new(extent.x, 0, 0),
            IVec3::new(0, extent.y, 0),
            IVec3::new(0, 0, extent.z),
            IVec3::new(extent.x, extent.y, 0),
            IVec3::new(extent.x, 0, extent.z),
            IVec3::new(0, extent.y, extent.z),
            extent,
        ];

        for corner in corners {
            let rotated_corner = orientation.mul_vec3(corner.as_vec3());
            let world_corner = *translation + rotated_corner.as_ivec3();

            min_extent = min_extent.min(world_corner);
            max_extent = max_extent.max(world_corner);
        }
    });

    let total_extent = UVec3::new(
        (max_extent.x - min_extent.x) as u32,
        (max_extent.y - min_extent.y) as u32,
        (max_extent.z - min_extent.z) as u32
    );

    let bricks_extent_x = ((total_extent.x) + BRICK_SIZE as u32 - 1) / BRICK_SIZE as u32; // ceiling
    let bricks_extent_y = ((total_extent.y) + BRICK_SIZE as u32 - 1) / BRICK_SIZE as u32;
    let bricks_extent_z = ((total_extent.z) + BRICK_SIZE as u32 - 1) / BRICK_SIZE as u32;
    
    let total_bricks = bricks_extent_x
        .saturating_mul(bricks_extent_y)
        .saturating_mul(bricks_extent_z) as u32;

    println!("min_extent: {}", min_extent);
    println!("max_extent: {}", max_extent);
    println!("total_extent (size): {}", total_extent);
    println!("bricks: {} x {} x {}", bricks_extent_x, bricks_extent_y, bricks_extent_z);

    let mut bricks: Vec<Brick> =
        vec![Brick { data: [0; BRICK_VOL] }; total_bricks as usize];

    let mut bytes = Vec::new();
    let magic = u32::from_ne_bytes(*b"brk\0");
    bytes.extend(&magic.to_ne_bytes());

    let position: [u32; 3] = [0, 0, 0];
    bytes.extend(bytemuck::cast_slice(&position));

    bytes.extend_from_slice(&total_extent.x.to_ne_bytes());
    bytes.extend_from_slice(&total_extent.y.to_ne_bytes());
    bytes.extend_from_slice(&total_extent.z.to_ne_bytes());

    // iterate through .vox scene models and their voxels
    // store voxel colours in each brick
    iterate_vox_tree(data, |model, translation, orientation| {
        for voxel in &model.voxels {
            let colour = &data.palette[voxel.i as usize];
            if colour.a == 0 { 
                continue; 
            }

            let voxel_value: u32 = ((colour.r as u32) << 24)
                | ((colour.g as u32) << 16)
                | ((colour.b as u32) << 8)
                | ((colour.a as u32) << 0);

            let local_voxel = IVec3::new(voxel.x as i32, voxel.y as i32, voxel.z as i32);

            let rotated_voxel = orientation.mul_vec3(local_voxel.as_vec3());

            let voxel_pos_i = *translation + rotated_voxel.as_ivec3() - min_extent;

            let b = (Vec3::new(voxel_pos_i.x as f32, voxel_pos_i.y as f32, voxel_pos_i.z as f32).floor() / BRICK_SIZE as f32).as_ivec3();
            let brick_i = b.as_uvec3();

            let in_brick_i = (voxel_pos_i - b * BRICK_SIZE as i32).as_uvec3();

            let brick_index = brick_i.x as usize
                + brick_i.y as usize * bricks_extent_x as usize
                + brick_i.z as usize * bricks_extent_x as usize * bricks_extent_y as usize;

            let in_brick_index = in_brick_i.x as usize
                + in_brick_i.y as usize * BRICK_SIZE
                + in_brick_i.z as usize * BRICK_SIZE * BRICK_SIZE;

            if brick_index < bricks.len() {
                bricks[brick_index].data[in_brick_index] = voxel_value;
            }
        }
    });

    let mut indices: Vec<u32> = vec![0; bricks.len()];
    for i in 0..bricks.len() {
        indices[i] = i as u32 | 0x80000000; // msb for loaded
    }
    bytes.extend(bytemuck::cast_slice(&indices));

    for brick in &bricks {
        bytes.extend(bytemuck::cast_slice(&brick.data));
    }

    bytes
}

// in progress/testing generation
/*pub fn generate_brickmap () -> GPUBrickMap {
    let brickmap = Box::new(VoxelMap {
        position: [0,0,0],
        extent: [CELL_SIZE as u32, CELL_SIZE as u32, CELL_SIZE as u32],
    });

    let mut indices: Vec<u32> = Vec::new();
    let mut bricks: Vec<Brick> = Vec::new();

    let magic = u32::from_ne_bytes(*b"brk\0");

    let perlin = Perlin::new(1);

    //brickmap.extent[0] = 8;
    //brickmap.extent[1] = 8;
    //brickmap.extent[2] = 8;

    let r: u32 = 255;
    let g: u32 = 156;
    let b: u32 = 68;
    let a: u32 = 255;

    indices.resize(CELL_SIZE * CELL_SIZE * CELL_SIZE, 0);
    for cell_x in 0..CELL_SIZE {
        for cell_y in 0..CELL_SIZE {
            for cell_z in 0..CELL_SIZE {
                let mut brick = Brick{data: [0; BRICK_VOL]};
                let mut loaded = true;
                let mut lod_mask = 0;

                let scale = 0.6;
                
                for x in 0..BRICK_SIZE {
                    for y in 0..BRICK_SIZE {
                        for z in 0..BRICK_SIZE {
                            let global_x = cell_x * BRICK_SIZE + x;
                            let global_y = cell_y * BRICK_SIZE + y;
                            let global_z = cell_z * BRICK_SIZE + z;

                            let height_noise = perlin.get([
                                global_x as f64 * scale,
                                global_z as f64 * scale
                            ]);

                            let terrain_height = ((height_noise + 1.0) * 0.5 * (CELL_SIZE as f64 / 2.0)) as i32;

                            if global_y <= (terrain_height as usize * BRICK_SIZE + x) {
                                let voxel_index = z + y * BRICK_SIZE + x * BRICK_SIZE * BRICK_SIZE;

                                
                                let voxel_value: u32 = ((x as u32) << 24) | ((y as u32) << 16) | ((z as u32) << 8) | (a << 0);
                                brick.data[voxel_index] = voxel_value;

                                loaded = true;
                                lod_mask |= 1 << (((x & 0b100) >> 2) + ((y & 0b100) >> 1) + (z & 0b100));
                            }
                        }
                    }
                }

                if loaded {
                    bricks.push(brick);
                    indices[cell_x + cell_y * CELL_SIZE + cell_z * CELL_SIZE * CELL_SIZE]
                        = (bricks.len() as u32 - 1) | 0x80000000 | lod_mask << 12;
                } else {
                    indices[cell_x + cell_y * CELL_SIZE + cell_z * CELL_SIZE * CELL_SIZE]
                        = (r << 16) | (g << 8) | (b << 0);
                }
            }
        }
    }
    
    let mut bytes = Vec::new();
    bytes.extend(&magic.to_ne_bytes());
    bytes.extend(bytemuck::cast_slice(&brickmap.position));
    bytes.extend(bytemuck::cast_slice(&brickmap.extent));
    bytes.extend(bytemuck::cast_slice(&indices));

    for brick in &bricks {
        bytes.extend(bytemuck::cast_slice(&brick.data));
    }

    GPUBrickMap { bytes: bytes }
}*/
