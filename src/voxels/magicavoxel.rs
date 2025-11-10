use dot_vox::*;
use glam::{IVec3, UVec3, Vec3, Mat3};

// edited from user https://github.com/entropylost
fn to_transform(position: IVec3, rotation: dot_vox::Rotation, size: UVec3) -> (Vec3, Mat3) {
    let position = position.as_vec3();

    let rotation = Mat3::from_cols_array_2d(&rotation.to_cols_array_2d());

    let mut offset = Vec3::select(
        (size % 2).cmpeq(UVec3::ZERO),
        Vec3::ZERO,
        Vec3::splat(0.5),
    );
    offset = rotation.mul_vec3(offset); // If another seam shows up in the future, try multiplying this with `scale`
    let center = rotation * (size.as_vec3() / 2.0);
    ((position - center + offset).into(), rotation.into())
}

// taken from dot_vox/examples/traverse_graph.rs
pub fn iterate_vox_tree(vox_tree: &DotVoxData, mut fun: impl FnMut(&Model, &IVec3, &Mat3)) {
    match &vox_tree.scenes[0] {
        SceneNode::Transform {
            attributes: _,
            frames: _,
            child,
            layer_id: _,
        } => {
            iterate_vox_tree_inner(
                vox_tree,
                *child,
                IVec3::new(0, 0, 0),
                Rotation::IDENTITY,
                &mut fun,
            );
        }
        _ => {
            panic!("The root node for a magicavoxel DAG should be a Transform node")
        }
    }
}

// edited from dot_vox/examples/traverse_graph.rs
fn iterate_vox_tree_inner(
    vox_tree: &DotVoxData,
    current_node: u32,
    translation: IVec3,
    rotation: Rotation,
    fun: &mut impl FnMut(&Model, &IVec3, &Mat3),
) {
    match &vox_tree.scenes[current_node as usize] {
        SceneNode::Transform {
            attributes: _,
            frames,
            child,
            layer_id: _,
        } => {
            // In case of a Transform node, the potential translation and rotation is added
            // to the global transform to all of the nodes children nodes
            let translation = if let Some(t) = frames[0].attributes.get("_t") {
                let translation_delta = t
                    .split(" ")
                    .map(|x| x.parse().expect("Not an integer!"))
                    .collect::<Vec<i32>>();
                debug_assert_eq!(translation_delta.len(), 3);
                translation
                    + IVec3::new(
                        translation_delta[0],
                        translation_delta[1],
                        translation_delta[2],
                    )
            } else {
                translation
            };
            let rotation = if let Some(r) = frames[0].attributes.get("_r") {
                rotation * Rotation::from_byte(
                    r.parse().expect("Expected valid u8 byte to parse rotation matrix"),
                )
            } else {
                Rotation::IDENTITY
            };

            iterate_vox_tree_inner(vox_tree, *child, translation, rotation, fun);
        }
        SceneNode::Group {
            attributes: _,
            children,
        } => {
            // in case the current node is a group, the index variable stores the current
            // child index
            for child_node in children {
                iterate_vox_tree_inner(vox_tree, *child_node, translation, rotation, fun);
            }
        }
        SceneNode::Shape {
            attributes: _,
            models,
        } => {
            // in case the current node is a shape: it's a leaf node and it contains
            // models(voxel arrays)
            for model in models {
                let model_size = vox_tree.models[model.model_id as usize].size;
                let (model_translation, model_rotation) = 
                    to_transform(translation, rotation, UVec3::new(model_size.x, model_size.y, model_size.z));
                fun(
                    &vox_tree.models[model.model_id as usize],
                    &model_translation.as_ivec3(),
                    &model_rotation,
                );
            }
        }
    }
}
