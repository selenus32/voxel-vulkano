
use noise::{NoiseFn, Perlin};
use std::io;
use std::io::prelude::*;
use std::fs::File;

const BRICK_SIZE: usize = 8;
const CELL_SIZE: usize = 64;
const BRICK_VOL: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;


struct Brick {
    data: [u32; BRICK_VOL],
}

pub struct GPUBrickMap {
    pub bytes: Vec<u8>,
}

struct BrickMap {
    magic: u32,
    position: [u32; 3],
    extent: [u32; 3],
    indices: Vec<u32>,
    bricks: Vec<Brick>,
}

pub fn read_brickmap (path: &str) -> io::Result<GPUBrickMap> {
    let mut f = File::open(path)?;
    let mut bytes = Vec::new();

    f.read_to_end(&mut bytes)?;

    Ok(GPUBrickMap {bytes: bytes})
}

pub fn generate_brickmap () -> GPUBrickMap {
    let mut brickmap = Box::new(BrickMap {
        magic: u32::from_ne_bytes(*b"brk\0"),
        position: [0,0,0],
        extent: [CELL_SIZE as u32, CELL_SIZE as u32, CELL_SIZE as u32],
        indices: Vec::new(),
        bricks: Vec::new(),
    });

    let perlin = Perlin::new(1);

    brickmap.extent[0] *= 8;
    brickmap.extent[1] *= 8;
    brickmap.extent[2] *= 8;

    brickmap.indices.resize(CELL_SIZE * CELL_SIZE * CELL_SIZE, 0);
    for cell_z in 0..CELL_SIZE {
        for cell_y in 0..CELL_SIZE {
            for cell_x in 0..CELL_SIZE {
                let mut brick = Brick{data: [0; BRICK_VOL]};
                let mut empty = true;
                let mut lod_mask = 0;

                let scale = 0.6;
                
                for z in 0..BRICK_SIZE {
                    for y in 0..BRICK_SIZE {
                        for x in 0..BRICK_SIZE {
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

                                let r: u32 = 255;
                                let g: u32 = 156;
                                let b: u32 = 68;
                                let a: u32 = 255;
                                let voxel_value: u32 = (r << 24) | (g << 16) | (b << 8) | (a << 0);
                                brick.data[voxel_index] = voxel_value;

                                empty = false;
                                lod_mask |= 1 << (((z & 0b100) >> 2) + ((y & 0b100) >> 1) + (x & 0b100));
                            }
                        }
                    }
                }

                if !empty {
                    brickmap.bricks.push(brick);
                    brickmap.indices[cell_z + cell_y * CELL_SIZE + cell_x * CELL_SIZE * CELL_SIZE]
                        = (brickmap.bricks.len() as u32 - 1) | 0x80000000 | (lod_mask << 12);
                }
            }
        }
    }
    
    let mut bytes = Vec::new();
    bytes.extend(&brickmap.magic.to_ne_bytes());
    bytes.extend(bytemuck::cast_slice(&brickmap.position));
    bytes.extend(bytemuck::cast_slice(&brickmap.extent));
    bytes.extend(bytemuck::cast_slice(&brickmap.indices));

    for brick in &brickmap.bricks {
        bytes.extend(bytemuck::cast_slice(&brick.data));
    }

    GPUBrickMap { bytes: bytes }
}
