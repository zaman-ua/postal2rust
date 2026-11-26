//! Shader compilation shared by the renderer and the passes of [`crate::dynamic`].

use glow::HasContext;

/// `source` with `#define NAME 1` after its version line.
pub(crate) fn with_define(source: &str, name: &str) -> String {
    source.replacen("#version 300 es\n", &format!("#version 300 es\n#define {name} 1\n"), 1)
}

/// Compiles and links a program; `what` names it in the panic message.
///
/// # Safety
/// A GL context must be current.
pub(crate) unsafe fn link(gl: &glow::Context, what: &str, vertex: &str, fragment: &str) -> glow::Program {
    unsafe {
        let program = gl.create_program().unwrap();
        for (kind, source) in [(glow::VERTEX_SHADER, vertex), (glow::FRAGMENT_SHADER, fragment)] {
            let shader = gl.create_shader(kind).unwrap();
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("{what} shader: {}", gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{what} link: {}", gl.get_program_info_log(program));
        }
        program
    }
}
