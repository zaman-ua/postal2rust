//! p2gles: thin OpenGL ES 3.0 renderer for POSTAL 2 (UE2-style baked lighting).
//! The `p2gles` binary is a standalone fly-through viewer; `p2game` can embed [`renderer::Renderer`].

pub mod renderer;
mod shaders;
