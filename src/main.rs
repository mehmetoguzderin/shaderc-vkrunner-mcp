use anyhow::Result;
use clap::Parser;
use image::codecs::pnm::PnmDecoder;
use image::{DynamicImage, ImageError, RgbImage};
use rmcp::{
    Error as McpError, RoleServer, ServerHandler, ServiceExt, const_string, model::*, schemars,
    service::RequestContext, tool, transport::stdio,
};
use serde_json::json;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::{self, EnvFilter};

pub fn read_and_decode_ppm_file<P: AsRef<Path>>(path: P) -> Result<RgbImage, ImageError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder: PnmDecoder<BufReader<File>> = PnmDecoder::new(reader)?;
    let dynamic_image = DynamicImage::from_decoder(decoder)?;
    let rgb_image = dynamic_image.into_rgb8();
    Ok(rgb_image)
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShaderStage {
    #[schemars(description = "Vertex processing stage (transforms vertices)")]
    Vert,
    #[schemars(description = "Fragment processing stage (determines pixel colors)")]
    Frag,
    #[schemars(description = "Tessellation control stage for patch control points")]
    Tesc,
    #[schemars(description = "Tessellation evaluation stage for computing tessellated geometry")]
    Tese,
    #[schemars(description = "Geometry stage for generating/modifying primitives")]
    Geom,
    #[schemars(description = "Compute stage for general-purpose computation")]
    Comp,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShaderRunnerRequire {
    #[schemars(
        description = "Enables cooperative matrix operations with specified dimensions and data type"
    )]
    CooperativeMatrix {
        m: u32,
        n: u32,
        component_type: String,
    },

    #[schemars(
        description = "Specifies depth/stencil format for depth testing and stencil operations"
    )]
    DepthStencil(String),

    #[schemars(description = "Specifies framebuffer format for render target output")]
    Framebuffer(String),

    #[schemars(description = "Enables double-precision floating point operations in shaders")]
    ShaderFloat64,

    #[schemars(description = "Enables geometry shader stage for primitive manipulation")]
    GeometryShader,

    #[schemars(description = "Enables lines with width greater than 1.0 pixel")]
    WideLines,

    #[schemars(description = "Enables bitwise logical operations on framebuffer contents")]
    LogicOp,

    #[schemars(description = "Specifies required subgroup size for shader execution")]
    SubgroupSize(u32),

    #[schemars(description = "Enables memory stores and atomic operations in fragment shaders")]
    FragmentStoresAndAtomics,

    #[schemars(description = "Enables shaders to use raw buffer addresses")]
    BufferDeviceAddress,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShaderRunnerPass {
    #[schemars(
        description = "Use a built-in pass-through vertex shader (forwards attributes to fragment shader)"
    )]
    VertPassthrough,

    #[schemars(description = "Use compiled vertex shader from specified SPIR-V assembly file")]
    VertSpirv {
        #[schemars(
            description = "Path to the compiled vertex shader SPIR-V assembly (.spvasm) file"
        )]
        vert_spvasm_path: String,
    },

    #[schemars(description = "Use compiled fragment shader from specified SPIR-V assembly file")]
    FragSpirv {
        #[schemars(
            description = "Path to the compiled fragment shader SPIR-V assembly (.spvasm) file"
        )]
        frag_spvasm_path: String,
    },

    #[schemars(description = "Use compiled compute shader from specified SPIR-V assembly file")]
    CompSpirv {
        #[schemars(
            description = "Path to the compiled compute shader SPIR-V assembly (.spvasm) file"
        )]
        comp_spvasm_path: String,
    },

    #[schemars(description = "Use compiled geometry shader from specified SPIR-V assembly file")]
    GeomSpirv {
        #[schemars(
            description = "Path to the compiled geometry shader SPIR-V assembly (.spvasm) file"
        )]
        geom_spvasm_path: String,
    },

    #[schemars(
        description = "Use compiled tessellation control shader from specified SPIR-V assembly file"
    )]
    TescSpirv {
        #[schemars(
            description = "Path to the compiled tessellation control shader SPIR-V assembly (.spvasm) file"
        )]
        tesc_spvasm_path: String,
    },

    #[schemars(
        description = "Use compiled tessellation evaluation shader from specified SPIR-V assembly file"
    )]
    TeseSpirv {
        #[schemars(
            description = "Path to the compiled tessellation evaluation shader SPIR-V assembly (.spvasm) file"
        )]
        tese_spvasm_path: String,
    },
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShaderRunnerVertexData {
    #[schemars(description = "Defines an attribute format at a shader location/binding")]
    AttributeFormat {
        #[schemars(description = "Location/binding number in the shader")]
        location: u32,

        #[schemars(description = "Format name (e.g., R32G32_SFLOAT, A8B8G8R8_UNORM_PACK32)")]
        format: String,
    },

    #[schemars(description = "2D position data (for R32G32_SFLOAT format)")]
    Vec2 {
        #[schemars(description = "X coordinate value")]
        x: f32,

        #[schemars(description = "Y coordinate value")]
        y: f32,
    },

    #[schemars(description = "3D position data (for R32G32B32_SFLOAT format)")]
    Vec3 {
        #[schemars(description = "X coordinate value")]
        x: f32,

        #[schemars(description = "Y coordinate value")]
        y: f32,

        #[schemars(description = "Z coordinate value")]
        z: f32,
    },

    #[schemars(description = "4D position/color data (for R32G32B32A32_SFLOAT format)")]
    Vec4 {
        #[schemars(description = "X coordinate or Red component")]
        x: f32,

        #[schemars(description = "Y coordinate or Green component")]
        y: f32,

        #[schemars(description = "Z coordinate or Blue component")]
        z: f32,

        #[schemars(description = "W coordinate or Alpha component")]
        w: f32,
    },

    #[schemars(description = "RGB color data (for R8G8B8_UNORM format)")]
    RGB {
        #[schemars(description = "Red component (0-255)")]
        r: u8,

        #[schemars(description = "Green component (0-255)")]
        g: u8,

        #[schemars(description = "Blue component (0-255)")]
        b: u8,
    },

    #[schemars(description = "ARGB color as hex (for A8B8G8R8_UNORM_PACK32 format)")]
    Hex {
        #[schemars(description = "Color as hex string (0xAARRGGBB format)")]
        value: String,
    },

    #[schemars(description = "Generic data components for custom formats")]
    GenericComponents {
        #[schemars(description = "Component values as strings, interpreted by format")]
        components: Vec<String>,
    },
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShaderRunnerTest {
    #[schemars(description = "Set fragment shader entrypoint function name")]
    FragmentEntrypoint {
        #[schemars(description = "Function name to use as entrypoint")]
        name: String,
    },

    #[schemars(description = "Set vertex shader entrypoint function name")]
    VertexEntrypoint {
        #[schemars(description = "Function name to use as entrypoint")]
        name: String,
    },

    #[schemars(description = "Set compute shader entrypoint function name")]
    ComputeEntrypoint {
        #[schemars(description = "Function name to use as entrypoint")]
        name: String,
    },

    #[schemars(description = "Set geometry shader entrypoint function name")]
    GeometryEntrypoint {
        #[schemars(description = "Function name to use as entrypoint")]
        name: String,
    },

    #[schemars(description = "Draw a rectangle using normalized device coordinates")]
    DrawRect {
        #[schemars(description = "X coordinate of lower-left corner")]
        x: f32,

        #[schemars(description = "Y coordinate of lower-left corner")]
        y: f32,

        #[schemars(description = "Width of the rectangle")]
        width: f32,

        #[schemars(description = "Height of the rectangle")]
        height: f32,
    },

    #[schemars(description = "Draw primitives using vertex data")]
    DrawArrays {
        #[schemars(description = "Primitive type (TRIANGLE_LIST, POINT_LIST, etc.)")]
        primitive_type: String,

        #[schemars(description = "Index of first vertex")]
        first: u32,

        #[schemars(description = "Number of vertices to draw")]
        count: u32,
    },

    #[schemars(description = "Draw primitives using indexed vertex data")]
    DrawArraysIndexed {
        #[schemars(description = "Primitive type (TRIANGLE_LIST, etc.)")]
        primitive_type: String,

        #[schemars(description = "Index of first index")]
        first: u32,

        #[schemars(description = "Number of indices to use")]
        count: u32,
    },

    #[schemars(description = "Create or initialize a Shader Storage Buffer Object (SSBO)")]
    SSBO {
        #[schemars(description = "Binding point in the shader")]
        binding: u32,

        #[schemars(description = "Size in bytes (if data not provided)")]
        size: Option<u32>,

        #[schemars(description = "Initial buffer contents")]
        data: Option<Vec<u8>>,

        #[schemars(description = "Descriptor set number (default: 0)")]
        descriptor_set: Option<u32>,
    },

    #[schemars(description = "Update a portion of an SSBO with new data")]
    SSBOSubData {
        #[schemars(description = "Binding point in the shader")]
        binding: u32,

        #[schemars(description = "Data type (float, vec4, etc.)")]
        data_type: String,

        #[schemars(description = "Byte offset into the buffer")]
        offset: u32,

        #[schemars(description = "Values to write (as strings)")]
        values: Vec<String>,

        #[schemars(description = "Descriptor set number (default: 0)")]
        descriptor_set: Option<u32>,
    },

    #[schemars(description = "Create or initialize a Uniform Buffer Object (UBO)")]
    UBO {
        #[schemars(description = "Binding point in the shader")]
        binding: u32,

        #[schemars(description = "Buffer contents")]
        data: Vec<u8>,

        #[schemars(description = "Descriptor set number (default: 0)")]
        descriptor_set: Option<u32>,
    },

    #[schemars(description = "Update a portion of a UBO with new data")]
    UBOSubData {
        #[schemars(description = "Binding point in the shader")]
        binding: u32,

        #[schemars(description = "Data type (float, vec4, etc.)")]
        data_type: String,

        #[schemars(description = "Byte offset into the buffer")]
        offset: u32,

        #[schemars(description = "Values to write (as strings)")]
        values: Vec<String>,

        #[schemars(description = "Descriptor set number (default: 0)")]
        descriptor_set: Option<u32>,
    },

    #[schemars(description = "Set memory layout for buffer data")]
    BufferLayout {
        #[schemars(description = "Buffer type (ubo, ssbo)")]
        buffer_type: String,

        #[schemars(description = "Layout specification (std140, std430, row_major, column_major)")]
        layout_type: String,
    },

    #[schemars(description = "Set push constant values")]
    Push {
        #[schemars(description = "Data type (float, vec4, etc.)")]
        data_type: String,

        #[schemars(description = "Byte offset into push constant block")]
        offset: u32,

        #[schemars(description = "Values to write (as strings)")]
        values: Vec<String>,
    },

    #[schemars(description = "Set memory layout for push constants")]
    PushLayout {
        #[schemars(description = "Layout specification (std140, std430)")]
        layout_type: String,
    },

    #[schemars(description = "Execute compute shader with specified workgroup counts")]
    Compute {
        #[schemars(description = "Number of workgroups in X dimension")]
        x: u32,

        #[schemars(description = "Number of workgroups in Y dimension")]
        y: u32,

        #[schemars(description = "Number of workgroups in Z dimension")]
        z: u32,
    },

    #[schemars(description = "Verify framebuffer or buffer contents match expected values")]
    Probe {
        #[schemars(description = "Probe type (all, rect, ssbo, etc.)")]
        probe_type: String,

        #[schemars(description = "Component format (rgba, rgb, etc.)")]
        format: String,

        #[schemars(description = "Parameters (coordinates, expected values)")]
        args: Vec<String>,
    },

    #[schemars(description = "Verify contents using normalized (0-1) coordinates")]
    RelativeProbe {
        #[schemars(description = "Probe type (rect, etc.)")]
        probe_type: String,

        #[schemars(description = "Component format (rgba, rgb, etc.)")]
        format: String,

        #[schemars(description = "Parameters (coordinates, expected values)")]
        args: Vec<String>,
    },

    #[schemars(description = "Set acceptable error margin for value comparisons")]
    Tolerance {
        #[schemars(description = "Error margins (absolute or percentage with % suffix)")]
        values: Vec<f32>,
    },

    #[schemars(description = "Clear the framebuffer to default values")]
    Clear,

    #[schemars(description = "Enable/disable depth testing")]
    DepthTestEnable {
        #[schemars(description = "True to enable depth testing")]
        enable: bool,
    },

    #[schemars(description = "Enable/disable writing to depth buffer")]
    DepthWriteEnable {
        #[schemars(description = "True to enable depth writes")]
        enable: bool,
    },

    #[schemars(description = "Set depth comparison function")]
    DepthCompareOp {
        #[schemars(description = "Function name (VK_COMPARE_OP_LESS, etc.)")]
        op: String,
    },

    #[schemars(description = "Enable/disable stencil testing")]
    StencilTestEnable {
        #[schemars(description = "True to enable stencil testing")]
        enable: bool,
    },

    #[schemars(description = "Define which winding order is front-facing")]
    FrontFace {
        #[schemars(description = "Mode (VK_FRONT_FACE_CLOCKWISE, etc.)")]
        mode: String,
    },

    #[schemars(description = "Configure stencil operation for a face")]
    StencilOp {
        #[schemars(description = "Face (front, back)")]
        face: String,

        #[schemars(description = "Operation name (passOp, failOp, etc.)")]
        op_name: String,

        #[schemars(description = "Value (VK_STENCIL_OP_REPLACE, etc.)")]
        value: String,
    },

    #[schemars(description = "Set reference value for stencil comparisons")]
    StencilReference {
        #[schemars(description = "Face (front, back)")]
        face: String,

        #[schemars(description = "Reference value")]
        value: u32,
    },

    #[schemars(description = "Set stencil comparison function")]
    StencilCompareOp {
        #[schemars(description = "Face (front, back)")]
        face: String,

        #[schemars(description = "Function (VK_COMPARE_OP_EQUAL, etc.)")]
        op: String,
    },

    #[schemars(description = "Control which color channels can be written")]
    ColorWriteMask {
        #[schemars(
            description = "Bit flags (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT, etc.)"
        )]
        mask: String,
    },

    #[schemars(description = "Enable/disable logical operations on colors")]
    LogicOpEnable {
        #[schemars(description = "True to enable logic operations")]
        enable: bool,
    },

    #[schemars(description = "Set type of logical operation on colors")]
    LogicOp {
        #[schemars(description = "Operation (VK_LOGIC_OP_XOR, etc.)")]
        op: String,
    },

    #[schemars(description = "Set face culling mode")]
    CullMode {
        #[schemars(description = "Mode (VK_CULL_MODE_BACK_BIT, etc.)")]
        mode: String,
    },

    #[schemars(description = "Set width for line primitives")]
    LineWidth {
        #[schemars(description = "Width in pixels")]
        width: f32,
    },

    #[schemars(description = "Specify a feature required by the test")]
    Require {
        #[schemars(description = "Feature name (subgroup_size, depthstencil, etc.)")]
        feature: String,

        #[schemars(description = "Feature parameters")]
        parameters: Vec<String>,
    },
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CompileRequest {
    #[schemars(description = "The shader stage to compile (vert, frag, comp, geom, tesc, tese)")]
    pub stage: ShaderStage,
    #[schemars(description = "GLSL shader source code to compile")]
    pub source: String,
    #[schemars(description = "Path where compiled SPIR-V assembly (.spvasm) will be saved")]
    pub tmp_output_path: String,
}
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CompileShadersRequest {
    #[schemars(description = "List of shader compile requests (produces SPIR-V assemblies)")]
    pub requests: Vec<CompileRequest>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CompileRunShadersRequest {
    #[schemars(
        description = "List of shader compile requests - each produces a SPIR-V assembly file"
    )]
    pub requests: Vec<CompileRequest>,
    #[schemars(description = "Optional hardware/feature requirements needed for shader execution")]
    pub requirements: Option<Vec<ShaderRunnerRequire>>,
    #[schemars(
        description = "Shader pipeline configuration (references compiled SPIR-V files by path)"
    )]
    pub passes: Vec<ShaderRunnerPass>,
    #[schemars(description = "Optional vertex data for rendering geometry")]
    pub vertex_data: Option<Vec<ShaderRunnerVertexData>>,
    #[schemars(description = "Test commands to execute (drawing, compute, verification, etc.)")]
    pub tests: Vec<ShaderRunnerTest>,
    #[schemars(description = "Optional path to save output image (PNG format)")]
    pub output_path: Option<String>,
}
#[derive(Clone)]
pub struct ShadercVkrunnerMcp {}
#[tool(tool_box)]
impl ShadercVkrunnerMcp {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    #[tool(
        description = "REQUIRES: (1) Shader compile requests that produce SPIR-V assembly files, (2) Passes referencing these files by path, and (3) Test commands for drawing/computation. Workflow: First compile GLSL to SPIR-V, then reference compiled files in passes, then execute drawing commands in tests to render/compute. Tests MUST include draw/compute commands to produce visible output. Optional: requirements for hardware features, vertex data for geometry, output path for saving image. Every shader MUST have its compiled output referenced in passes."
    )]
    fn compile_run_shaders(
        &self,
        #[tool(aggr)] request: CompileRunShadersRequest,
    ) -> Result<CallToolResult, McpError> {
        use std::fs::File;
        use std::io::{Read, Write};
        use std::path::Path;
        use std::process::{Command, Stdio};

        fn io_err(e: std::io::Error) -> McpError {
            McpError::internal_error("IO operation failed", Some(json!({"error": e.to_string()})))
        }

        for req in &request.requests {
            let stage_flag = match req.stage {
                ShaderStage::Vert => "vert",
                ShaderStage::Frag => "frag",
                ShaderStage::Tesc => "tesc",
                ShaderStage::Tese => "tese",
                ShaderStage::Geom => "geom",
                ShaderStage::Comp => "comp",
            };

            let tmp_output_path = if req.tmp_output_path.starts_with("/tmp") {
                req.tmp_output_path.clone()
            } else {
                format!("/tmp/{}", req.tmp_output_path)
            };

            if let Some(parent) = Path::new(&tmp_output_path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    McpError::internal_error(
                        "Failed to create temporary output directory",
                        Some(json!({"error": e.to_string()})),
                    )
                })?;
            }

            let mut child = Command::new("glslc")
                .arg("--target-env=vulkan1.4")
                .arg(format!("-fshader-stage={stage_flag}"))
                .arg("-O")
                .arg("-S")
                .arg("-o")
                .arg(&tmp_output_path)
                .arg("-")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    McpError::internal_error(
                        "Failed to spawn glslc process",
                        Some(json!({"error": e.to_string()})),
                    )
                })?;

            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(req.source.as_bytes()).map_err(|e| {
                    McpError::internal_error(
                        "Failed to write to glslc stdin",
                        Some(json!({"error": e.to_string()})),
                    )
                })?;
            }

            let output = child.wait_with_output().map_err(|e| {
                McpError::internal_error(
                    "Failed to wait for glslc process",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Shader compilation failed:\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}"
                ))]));
            }
        }

        let shader_test_path = "/tmp/vkrunner_test.shader_test";
        let mut shader_test_file = File::create(shader_test_path).map_err(io_err)?;

        if let Some(requirements) = &request.requirements {
            if !requirements.is_empty() {
                writeln!(shader_test_file, "[require]").map_err(io_err)?;

                for req in requirements {
                    match req {
                        ShaderRunnerRequire::CooperativeMatrix {
                            m,
                            n,
                            component_type,
                        } => {
                            writeln!(
                                shader_test_file,
                                "cooperative_matrix m={m} n={n} c={component_type}"
                            )
                            .map_err(io_err)?;
                        }
                        ShaderRunnerRequire::DepthStencil(format) => {
                            writeln!(shader_test_file, "depthstencil {format}").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::Framebuffer(format) => {
                            writeln!(shader_test_file, "framebuffer {format}").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::ShaderFloat64 => {
                            writeln!(shader_test_file, "shaderFloat64").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::GeometryShader => {
                            writeln!(shader_test_file, "geometryShader").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::WideLines => {
                            writeln!(shader_test_file, "wideLines").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::LogicOp => {
                            writeln!(shader_test_file, "logicOp").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::SubgroupSize(size) => {
                            writeln!(shader_test_file, "subgroup_size {size}").map_err(io_err)?;
                        }
                        ShaderRunnerRequire::FragmentStoresAndAtomics => {
                            writeln!(shader_test_file, "fragmentStoresAndAtomics")
                                .map_err(io_err)?;
                        }
                        ShaderRunnerRequire::BufferDeviceAddress => {
                            writeln!(shader_test_file, "bufferDeviceAddress").map_err(io_err)?;
                        }
                    }
                }

                writeln!(shader_test_file).map_err(io_err)?;
            }
        }

        for pass in &request.passes {
            match pass {
                ShaderRunnerPass::VertPassthrough => {
                    writeln!(shader_test_file, "[vertex shader passthrough]").map_err(io_err)?;
                }
                ShaderRunnerPass::VertSpirv { vert_spvasm_path } => {
                    writeln!(shader_test_file, "[vertex shader spirv]").map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if vert_spvasm_path.starts_with("/tmp") {
                        vert_spvasm_path.clone()
                    } else {
                        format!("/tmp/{vert_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to open vertex shader SPIR-V file at {path}"),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read vertex shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
                ShaderRunnerPass::FragSpirv { frag_spvasm_path } => {
                    writeln!(shader_test_file, "[fragment shader spirv]").map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if frag_spvasm_path.starts_with("/tmp") {
                        frag_spvasm_path.clone()
                    } else {
                        format!("/tmp/{frag_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to open fragment shader SPIR-V file at {path}"),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read fragment shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
                ShaderRunnerPass::CompSpirv { comp_spvasm_path } => {
                    writeln!(shader_test_file, "[compute shader spirv]").map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if comp_spvasm_path.starts_with("/tmp") {
                        comp_spvasm_path.clone()
                    } else {
                        format!("/tmp/{comp_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to open compute shader SPIR-V file at {path}"),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read compute shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
                ShaderRunnerPass::GeomSpirv { geom_spvasm_path } => {
                    writeln!(shader_test_file, "[geometry shader spirv]").map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if geom_spvasm_path.starts_with("/tmp") {
                        geom_spvasm_path.clone()
                    } else {
                        format!("/tmp/{geom_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to open geometry shader SPIR-V file at {path}"),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read geometry shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
                ShaderRunnerPass::TescSpirv { tesc_spvasm_path } => {
                    writeln!(shader_test_file, "[tessellation control shader spirv]")
                        .map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if tesc_spvasm_path.starts_with("/tmp") {
                        tesc_spvasm_path.clone()
                    } else {
                        format!("/tmp/{tesc_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!(
                                    "Failed to open tessellation control shader SPIR-V file at {path}"
                                ),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read tessellation control shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
                ShaderRunnerPass::TeseSpirv { tese_spvasm_path } => {
                    writeln!(shader_test_file, "[tessellation evaluation shader spirv]")
                        .map_err(io_err)?;

                    let mut spvasm = String::new();
                    let path = if tese_spvasm_path.starts_with("/tmp") {
                        tese_spvasm_path.clone()
                    } else {
                        format!("/tmp/{tese_spvasm_path}")
                    };

                    File::open(&path)
                        .map_err(|e| {
                            McpError::internal_error(
                                format!(
                                    "Failed to open tessellation evaluation shader SPIR-V file at {path}"
                                ),
                                Some(json!({"error": e.to_string()})),
                            )
                        })?
                        .read_to_string(&mut spvasm)
                        .map_err(|e| {
                            McpError::internal_error(
                                "Failed to read tessellation evaluation shader SPIR-V file",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;
                    writeln!(shader_test_file, "{spvasm}").map_err(io_err)?;
                }
            }

            writeln!(shader_test_file).map_err(io_err)?;
        }

        if let Some(vertex_data) = &request.vertex_data {
            writeln!(shader_test_file, "[vertex data]").map_err(io_err)?;

            for data in vertex_data {
                if let ShaderRunnerVertexData::AttributeFormat { location, format } = data {
                    writeln!(shader_test_file, "{location}/{format}").map_err(io_err)?;
                } else {
                    match data {
                        ShaderRunnerVertexData::Vec2 { x, y } => {
                            write!(shader_test_file, "{x} {y}").map_err(io_err)?;
                        }
                        ShaderRunnerVertexData::Vec3 { x, y, z } => {
                            write!(shader_test_file, "{x} {y} {z}").map_err(io_err)?;
                        }
                        ShaderRunnerVertexData::Vec4 { x, y, z, w } => {
                            write!(shader_test_file, "{x} {y} {z} {w}").map_err(io_err)?;
                        }
                        ShaderRunnerVertexData::RGB { r, g, b } => {
                            write!(shader_test_file, "{r} {g} {b}").map_err(io_err)?;
                        }
                        ShaderRunnerVertexData::Hex { value } => {
                            write!(shader_test_file, "{value}").map_err(io_err)?;
                        }
                        ShaderRunnerVertexData::GenericComponents { components } => {
                            for component in components {
                                write!(shader_test_file, "{component} ").map_err(io_err)?;
                            }
                        }
                        _ => {}
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
            }

            writeln!(shader_test_file).map_err(io_err)?;
        }

        writeln!(shader_test_file, "[test]").map_err(io_err)?;

        for test_cmd in &request.tests {
            match test_cmd {
                ShaderRunnerTest::FragmentEntrypoint { name } => {
                    writeln!(shader_test_file, "fragment entrypoint {name}").map_err(io_err)?;
                }
                ShaderRunnerTest::VertexEntrypoint { name } => {
                    writeln!(shader_test_file, "vertex entrypoint {name}").map_err(io_err)?;
                }
                ShaderRunnerTest::ComputeEntrypoint { name } => {
                    writeln!(shader_test_file, "compute entrypoint {name}").map_err(io_err)?;
                }
                ShaderRunnerTest::GeometryEntrypoint { name } => {
                    writeln!(shader_test_file, "geometry entrypoint {name}").map_err(io_err)?;
                }
                ShaderRunnerTest::DrawRect {
                    x,
                    y,
                    width,
                    height,
                } => {
                    writeln!(shader_test_file, "draw rect {x} {y} {width} {height}")
                        .map_err(io_err)?;
                }
                ShaderRunnerTest::DrawArrays {
                    primitive_type,
                    first,
                    count,
                } => {
                    writeln!(
                        shader_test_file,
                        "draw arrays {primitive_type} {first} {count}"
                    )
                    .map_err(io_err)?;
                }
                ShaderRunnerTest::DrawArraysIndexed {
                    primitive_type,
                    first,
                    count,
                } => {
                    writeln!(
                        shader_test_file,
                        "draw arrays indexed {primitive_type} {first} {count}"
                    )
                    .map_err(io_err)?;
                }
                ShaderRunnerTest::SSBO {
                    binding,
                    size,
                    data,
                    descriptor_set,
                } => {
                    let set_prefix = if let Some(set) = descriptor_set {
                        format!("{set}:")
                    } else {
                        String::new()
                    };

                    if let Some(size) = size {
                        writeln!(shader_test_file, "ssbo {set_prefix}{binding} {size}")
                            .map_err(io_err)?;
                    } else if let Some(_data) = data {
                        writeln!(shader_test_file, "ssbo {set_prefix}{binding} data")
                            .map_err(io_err)?;
                    }
                }
                ShaderRunnerTest::SSBOSubData {
                    binding,
                    data_type,
                    offset,
                    values,
                    descriptor_set,
                } => {
                    let set_prefix = if let Some(set) = descriptor_set {
                        format!("{set}:")
                    } else {
                        String::new()
                    };

                    write!(
                        shader_test_file,
                        "ssbo {set_prefix}{binding} subdata {data_type} {offset}"
                    )
                    .map_err(io_err)?;
                    for value in values {
                        write!(shader_test_file, " {value}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::UBO {
                    binding,
                    data: _,
                    descriptor_set,
                } => {
                    let set_prefix = if let Some(set) = descriptor_set {
                        format!("{set}:")
                    } else {
                        String::new()
                    };

                    writeln!(shader_test_file, "ubo {set_prefix}{binding} data").map_err(io_err)?;
                }
                ShaderRunnerTest::UBOSubData {
                    binding,
                    data_type,
                    offset,
                    values,
                    descriptor_set,
                } => {
                    let set_prefix = if let Some(set) = descriptor_set {
                        format!("{set}:")
                    } else {
                        String::new()
                    };

                    write!(
                        shader_test_file,
                        "ubo {set_prefix}{binding} subdata {data_type} {offset}"
                    )
                    .map_err(io_err)?;
                    for value in values {
                        write!(shader_test_file, " {value}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::BufferLayout {
                    buffer_type,
                    layout_type,
                } => {
                    writeln!(shader_test_file, "{buffer_type} layout {layout_type}")
                        .map_err(io_err)?;
                }
                ShaderRunnerTest::Push {
                    data_type,
                    offset,
                    values,
                } => {
                    write!(shader_test_file, "push {data_type} {offset}").map_err(io_err)?;
                    for value in values {
                        write!(shader_test_file, " {value}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::PushLayout { layout_type } => {
                    writeln!(shader_test_file, "push layout {layout_type}").map_err(io_err)?;
                }
                ShaderRunnerTest::Compute { x, y, z } => {
                    writeln!(shader_test_file, "compute {x} {y} {z}").map_err(io_err)?;
                }
                ShaderRunnerTest::Probe {
                    probe_type,
                    format,
                    args,
                } => {
                    write!(shader_test_file, "probe {probe_type} {format}").map_err(io_err)?;
                    for arg in args {
                        write!(shader_test_file, " {arg}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::RelativeProbe {
                    probe_type,
                    format,
                    args,
                } => {
                    write!(shader_test_file, "relative probe {probe_type} {format}")
                        .map_err(io_err)?;
                    for arg in args {
                        write!(shader_test_file, " {arg}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::Tolerance { values } => {
                    write!(shader_test_file, "tolerance").map_err(io_err)?;
                    for value in values {
                        write!(shader_test_file, " {value}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
                ShaderRunnerTest::Clear => {
                    writeln!(shader_test_file, "clear").map_err(io_err)?;
                }
                ShaderRunnerTest::DepthTestEnable { enable } => {
                    writeln!(shader_test_file, "depthTestEnable {enable}").map_err(io_err)?;
                }
                ShaderRunnerTest::DepthWriteEnable { enable } => {
                    writeln!(shader_test_file, "depthWriteEnable {enable}").map_err(io_err)?;
                }
                ShaderRunnerTest::DepthCompareOp { op } => {
                    writeln!(shader_test_file, "depthCompareOp {op}").map_err(io_err)?;
                }
                ShaderRunnerTest::StencilTestEnable { enable } => {
                    writeln!(shader_test_file, "stencilTestEnable {enable}").map_err(io_err)?;
                }
                ShaderRunnerTest::FrontFace { mode } => {
                    writeln!(shader_test_file, "frontFace {mode}").map_err(io_err)?;
                }
                ShaderRunnerTest::StencilOp {
                    face,
                    op_name,
                    value,
                } => {
                    writeln!(shader_test_file, "{face}.{op_name} {value}",).map_err(io_err)?;
                }
                ShaderRunnerTest::StencilReference { face, value } => {
                    writeln!(shader_test_file, "{face}.reference {value}",).map_err(io_err)?;
                }
                ShaderRunnerTest::StencilCompareOp { face, op } => {
                    writeln!(shader_test_file, "{face}.compareOp {op}",).map_err(io_err)?;
                }
                ShaderRunnerTest::ColorWriteMask { mask } => {
                    writeln!(shader_test_file, "colorWriteMask {mask}",).map_err(io_err)?;
                }
                ShaderRunnerTest::LogicOpEnable { enable } => {
                    writeln!(shader_test_file, "logicOpEnable {enable}",).map_err(io_err)?;
                }
                ShaderRunnerTest::LogicOp { op } => {
                    writeln!(shader_test_file, "logicOp {op}",).map_err(io_err)?;
                }
                ShaderRunnerTest::CullMode { mode } => {
                    writeln!(shader_test_file, "cullMode {mode}",).map_err(io_err)?;
                }
                ShaderRunnerTest::LineWidth { width } => {
                    writeln!(shader_test_file, "lineWidth {width}").map_err(io_err)?;
                }
                ShaderRunnerTest::Require {
                    feature,
                    parameters,
                } => {
                    write!(shader_test_file, "require {feature}").map_err(io_err)?;
                    for param in parameters {
                        write!(shader_test_file, " {param}").map_err(io_err)?;
                    }
                    writeln!(shader_test_file).map_err(io_err)?;
                }
            }
        }

        shader_test_file.flush().map_err(io_err)?;

        let tmp_image_path = "/tmp/vkrunner_output.ppm";
        if let Some(output_path) = &request.output_path {
            if output_path.starts_with("/tmp") {
                tmp_image_path.to_string()
            } else {
                format!("/tmp/{output_path}")
            }
        } else {
            tmp_image_path.to_string()
        };
        let mut vkrunner_args = vec![shader_test_path];

        if request.output_path.is_some() {
            vkrunner_args.push("--image");
            vkrunner_args.push(tmp_image_path);
        }

        let vkrunner_output = Command::new("vkrunner")
            .args(&vkrunner_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| {
                McpError::internal_error(
                    "Failed to run vkrunner",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

        let stdout = String::from_utf8_lossy(&vkrunner_output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&vkrunner_output.stderr).to_string();

        let mut result_message = if vkrunner_output.status.success() {
            format!("VkRunner execution successful.\n\nOutput:\n{stdout}\n\n")
        } else {
            format!("VkRunner execution failed.\n\nOutput:\n{stdout}\n\nError:\n{stderr}\n\n",)
        };

        if let Some(output_path) = &request.output_path {
            if vkrunner_output.status.success() && Path::new(tmp_image_path).exists() {
                match read_and_decode_ppm_file(tmp_image_path) {
                    Ok(img) => {
                        if let Some(parent) = Path::new(output_path).parent() {
                            if !parent.as_os_str().is_empty() {
                                std::fs::create_dir_all(parent).map_err(|e| {
                                    McpError::internal_error(
                                        "Failed to create output directory",
                                        Some(json!({"error": e.to_string()})),
                                    )
                                })?;
                            }
                        }

                        img.save(output_path).map_err(|e| {
                            McpError::internal_error(
                                "Failed to save output image",
                                Some(json!({"error": e.to_string()})),
                            )
                        })?;

                        result_message.push_str(&format!("Image saved to: {output_path}\n"));
                    }
                    Err(e) => {
                        result_message.push_str(&format!("Failed to convert output image: {e}\n"));
                    }
                }
            } else if vkrunner_output.status.success() {
                result_message.push_str("No output image was generated by VkRunner.\n");
            }
        }

        result_message.push_str("\nShader Test File Contents:\n");
        result_message.push_str(
            &std::fs::read_to_string(shader_test_path)
                .unwrap_or_else(|_| "Failed to read shader test file".to_string()),
        );

        Ok(CallToolResult::success(vec![Content::text(result_message)]))
    }
}

impl Default for ShadercVkrunnerMcp {
    fn default() -> Self {
        Self::new()
    }
}

const_string!(Echo = "echo");
#[tool(tool_box)]
impl ServerHandler for ShadercVkrunnerMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server provides tools for compiling and running GLSL shaders using Vulkan infrastructure. The typical workflow is:

            1. Compile GLSL source code to SPIR-V assembly using the 'compile_run_shaders' tool
            2. Specify hardware/feature requirements if needed (e.g., geometry shaders, floating-point formats)
            3. Reference compiled SPIR-V files in shader passes
            4. Define vertex data if rendering geometry
            5. Set up test commands to draw or compute
            6. Optionally save the rendered output as an image".to_string()),
        }
    }
}

#[derive(Parser, Debug)]
#[command(about)]
struct Args {
    #[clap(short, long, value_parser)]
    work_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(work_dir) = args.work_dir {
        std::env::set_current_dir(&work_dir)
            .map_err(|e| {
                eprintln!("Failed to set working directory to {work_dir:?}: {e}");
                std::process::exit(1);
            })
            .unwrap();
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting MCP server");

    let service = ShadercVkrunnerMcp::new()
        .serve(stdio())
        .await
        .inspect_err(|e| {
            tracing::error!("serving error: {:?}", e);
        })?;

    service.waiting().await?;
    Ok(())
}
