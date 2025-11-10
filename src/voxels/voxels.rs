use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use crate::voxels::brickmap::*;

pub fn get_voxel_bytes(model_filepath: &str) -> Vec<u8> {
    let mut f = File::open(model_filepath).expect("failed to open model file");
    let mut bytes = Vec::new();

    f.read_to_end(&mut bytes).expect("failed to read model file to end");

    bytes
}

pub fn read_voxel_model (model_filepath: &str) -> Vec<u8> {
    let path = Path::new(model_filepath);
    let format = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");

    match format {
        "vox" => {
            println!("Parsing magicavoxel file");
            //let vox_bytes = get_voxel_bytes(model_filepath)?;
            let result = dot_vox::load(model_filepath)
                .ok()
                .expect("expected a valid .vox file");

            let brk_bytes = parse_vox_to_brk(&result);
            //Ok(brk_bytes)
            
            brk_bytes
        },
        "brk" => {
            println!("Parsing brickmap file");
            get_voxel_bytes(model_filepath)
        },
        "" => {
            println!("No usable voxel format given");
            Vec::new()
        }
        _ => {
            println!("No usable voxel format given");
            Vec::new()
        }
    }
}