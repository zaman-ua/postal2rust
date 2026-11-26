//! GL smoke test of the generic pass (`dynamic.rs`) without game files: compiles every shader and
//! draws one mesh per path (skinned, instanced, canvas, particles, grade), then checks glGetError.
//! Run: `xvfb-run -a cargo run --release -p p2gles --example smoke`

use bevy_math::{Mat4, Vec3};
use glow::HasContext;
use glutin::config::{Api, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin_winit::{DisplayBuilder, GlWindow};
use p2gles::dynamic::*;
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App;

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_inner_size(PhysicalSize::new(320, 240));
        let template = ConfigTemplateBuilder::new().with_api(Api::GLES3).with_depth_size(24).with_alpha_size(0);
        let (window, config) = DisplayBuilder::new().with_preference(glutin_winit::ApiPreference::PreferEgl).with_window_attributes(Some(attrs)).build(el, template, |c| c.reduce(|a, b| if b.num_samples() < a.num_samples() { b } else { a }).unwrap()).expect("display");
        let window = window.unwrap();
        let raw = window.window_handle().unwrap().as_raw();
        let display = config.display();
        let cattrs = ContextAttributesBuilder::new().with_context_api(ContextApi::Gles(Some(Version::new(3, 0)))).build(Some(raw));
        let ctx = unsafe { display.create_context(&config, &cattrs).expect("ctx") };
        let sattrs = window.build_surface_attributes(Default::default()).unwrap();
        let surface = unsafe { display.create_window_surface(&config, &sattrs).unwrap() };
        let ctx = ctx.make_current(&surface).unwrap();
        let gl = unsafe { glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s)) };
        println!("GL {} / {}", unsafe { gl.get_parameter_string(glow::VERSION) }, unsafe { gl.get_parameter_string(glow::RENDERER) });
        let mut d = Dynamic::new(&gl);
        let pos = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]];
        let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let col = [[1.0f32; 4]; 4];
        let joints = [[0u16, 1, 0, 0]; 4];
        let weights = [[0.5f32, 0.5, 0.0, 0.0]; 4];
        let idx = [0u32, 1, 2, 0, 2, 3];
        let mesh = |j: bool| MeshIn { pos: &pos, normal: None, uv0: Some(&uv), uv1: Some(&uv), color: Some(&col), joints: j.then_some(&joints[..]), weights: j.then_some(&weights[..]), indices: &idx };
        d.add_mesh(&gl, 1, &mesh(false));
        d.add_mesh(&gl, 2, &mesh(true));
        let px: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 7 % 255) as u8).collect();
        d.add_tex(&gl, 10, &TexIn { width: 16, height: 16, format: p2view::scene::PixelFormat::Rgba8, mips: vec![&px], repeat: true });
        let base = Draw { mesh: 1, tex: Some(10), lightmap: Some(10), model: Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)), joints: None, color: [1.0; 4], blend: Blend::Opaque, cutoff: 0.5, two_sided: false, light: Light::Dynamic, uv: UV_IDENTITY, depth: 5.0, overlay: false };
        let mut draws = vec![base.clone()];
        for l in [Light::Lightmap, Light::Vertex, Light::Unlit, Light::Unlit2x] {
            draws.push(Draw { light: l, ..base.clone() });
        }
        for b in [Blend::Mask, Blend::Alpha, Blend::Add, Blend::Multiply] {
            draws.push(Draw { blend: b, ..base.clone() });
        }
        draws.push(Draw { mesh: 2, joints: Some(std::sync::Arc::new(vec![Mat4::IDENTITY; 3])), ..base.clone() });
        for i in 0..5 {
            draws.push(Draw { model: Mat4::from_translation(Vec3::new(i as f32, 0.0, -6.0)), blend: Blend::Mask, ..base.clone() }); // instanced run
        }
        draws.push(Draw { overlay: true, ..base.clone() });
        let env = Env { fog_start: 1.0, fog_end: 50.0, fog_color: [0.5; 3], ambient: [0.5; 3], sun_dir: Vec3::NEG_Y, sun_color: [0.5; 3] };
        let vp = Mat4::perspective_rh_gl(1.2, 320.0 / 240.0, 0.1, 100.0);
        unsafe {
            gl.viewport(0, 0, 320, 240);
            gl.clear_color(0.2, 0.3, 0.4, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        }
        d.render(&gl, vp, Vec3::ZERO, &env, &mut draws);
        let mut parts: Vec<ParticleDraw> = (0..7).map(|s| ParticleDraw { mesh: 1, tex: Some(10), style: s, z_test: true, z_write: false, alpha_ref: 0.1, alpha_test: s == 0, order: s as f32 }).collect();
        d.render_particles(&gl, vp, Vec3::ZERO, &env, &mut parts);
        let cd = |b| CanvasDraw { mesh: 1, tex: [Some(10), None, None], tint: [1.0; 4], slot_color: [[1.0; 4]; 3], xu: [[1.0, 0.0, 0.0, 0.0]; 3], xv: [[0.0, 1.0, 0.0, 0.0]; 3], kinds: [1, 0, 0], ops: [0, 0, 0, 0], modulate: 1.0, alpha_test: -1.0, blend: b };
        let canvas: Vec<CanvasDraw> = [CanvasBlend::Opaque, CanvasBlend::Alpha, CanvasBlend::Translucent, CanvasBlend::Modulate, CanvasBlend::AlphaModulate, CanvasBlend::Darken, CanvasBlend::Brighten, CanvasBlend::Invisible].into_iter().map(cd).collect();
        d.render_canvas(&gl, (320, 240), &canvas);
        d.render_grade(&gl, (320, 240), 0.1, 1.2, 0.9);
        let mut buf = vec![0u8; 320 * 240 * 4];
        unsafe {
            gl.read_pixels(0, 0, 320, 240, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut buf)));
        }
        let err = unsafe { gl.get_error() };
        let nonbg = buf.chunks(4).filter(|p| p[..3] != [51, 76, 102]).count();
        println!("draws {} tris {}  glError {err:#x}  non-background pixels {nonbg}", d.draws, d.tris);
        let _ = (surface, ctx);
        std::process::exit(if err == 0 && nonbg > 0 { 0 } else { 1 });
    }
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: winit::event::WindowEvent) {}
}

fn main() {
    EventLoop::new().unwrap().run_app(&mut App).unwrap();
}
