package codegen

import (
	"testing"
)

// The SPIR-V backend had no case for ir.ExprRelational, so `any()` and `all()`
// over a vector of bools — the natural way to write a rect rejection in a
// material — parsed, validated, lowered, and then died at pipeline creation with
// "unsupported expression kind: ir.ExprRelational". Every other backend lowered
// it; only the Vulkan path failed, and only once a pipeline was actually built.
//
// These tests pin the whole family: any/all over vec2/vec3/vec4<bool>, the
// scalar-bool form that has no vector to reduce, and isnan/isinf which keep the
// vector shape instead of collapsing it.

// relationalRectShader is the shape from the issue: a frame-rect rejection in a
// fragment shader, written with any() instead of spelled out component-wise.
const relationalRectShader = `
struct Frame { rect: vec4<f32> }
@group(0) @binding(0) var<uniform> frame: Frame;

@fragment
fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = p.xy;
    if (any(uv < frame.rect.xy) || any(uv > frame.rect.zw)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`

func TestRelationalRectRejectionCompilesToSPIRV(t *testing.T) {
	spirvBytes := compileRelationalShader(t, "rect", relationalRectShader)
	assertHasOpcode(t, spirvBytes, OpAny, "any() over a vec2<bool>")
}

// TestRelationalAnyAllOverVectors covers both reductions at every vector width.
func TestRelationalAnyAllOverVectors(t *testing.T) {
	cases := []struct {
		name    string
		fun     string
		size    string
		swizzle string
		opcode  OpCode
	}{
		{"any/vec2", "any", "vec2<f32>", ".xy", OpAny},
		{"any/vec3", "any", "vec3<f32>", ".xyz", OpAny},
		{"any/vec4", "any", "vec4<f32>", "", OpAny},
		{"all/vec2", "all", "vec2<f32>", ".xy", OpAll},
		{"all/vec3", "all", "vec3<f32>", ".xyz", OpAll},
		{"all/vec4", "all", "vec4<f32>", "", OpAll},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := `
@fragment
fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let v = p` + tc.swizzle + `;
    let hit = ` + tc.fun + `(v > ` + tc.size + `(0.5));
    return select(vec4<f32>(0.0), vec4<f32>(1.0), hit);
}
`
			spirvBytes := compileRelationalShader(t, tc.name, src)
			assertHasOpcode(t, spirvBytes, tc.opcode, tc.fun+" over "+tc.size)
		})
	}
}

// TestRelationalAnyAllOverScalarBool covers the degenerate form. SPIR-V's OpAny
// and OpAll take a *vector* of bools, so a scalar argument has nothing to reduce
// and must pass straight through rather than emit an invalid instruction.
func TestRelationalAnyAllOverScalarBool(t *testing.T) {
	for _, fun := range []string{"any", "all"} {
		t.Run(fun, func(t *testing.T) {
			src := `
@fragment
fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let hit = ` + fun + `(p.x > 0.5);
    return select(vec4<f32>(0.0), vec4<f32>(1.0), hit);
}
`
			spirvBytes := compileRelationalShader(t, fun+"/scalar", src)
			assertLacksOpcode(t, spirvBytes, OpAny, fun+" over a scalar bool")
			assertLacksOpcode(t, spirvBytes, OpAll, fun+" over a scalar bool")
		})
	}
}

// TestRelationalIsNanIsInf covers the two that keep the argument's shape: over a
// vector they produce a vec<bool>, not a single bool.
func TestRelationalIsNanIsInf(t *testing.T) {
	cases := []struct {
		name   string
		fun    string
		arg    string
		opcode OpCode
	}{
		{"isnan/scalar", "isnan", "p.x", OpIsNan},
		{"isnan/vector", "isnan", "p.xy", OpIsNan},
		{"isinf/scalar", "isinf", "p.x", OpIsInf},
		{"isinf/vector", "isinf", "p.xy", OpIsInf},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := `
@fragment
fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let bad = any(vec2<bool>(` + tc.fun + `(` + tc.arg + `)));
    return select(vec4<f32>(0.0), vec4<f32>(1.0), bad);
}
`
			spirvBytes := compileRelationalShader(t, tc.name, src)
			assertHasOpcode(t, spirvBytes, tc.opcode, tc.fun+"("+tc.arg+")")
		})
	}
}

// compileRelationalShader compiles a shader and runs spirv-val over the result.
// The bug this file pins never showed up in parsing or validation — it surfaced
// only once something actually consumed the SPIR-V — so these tests check the
// binary a driver would be handed, not just that compilation returned.
func compileRelationalShader(t *testing.T, label, source string) []byte {
	t.Helper()
	spirvBytes := compileWGSLToSPIRV(t, label, source)
	validateWithVulkanSDK(t, spirvBytes)
	return spirvBytes
}

func assertHasOpcode(t *testing.T, spirvBytes []byte, op OpCode, what string) {
	t.Helper()
	if !hasOpcode(spirvBytes, op) {
		t.Fatalf("%s emitted no %v\n%s", what, op, disassembleSPIRV(spirvBytes))
	}
}

func assertLacksOpcode(t *testing.T, spirvBytes []byte, op OpCode, what string) {
	t.Helper()
	if hasOpcode(spirvBytes, op) {
		t.Fatalf("%s emitted %v, which requires a vector operand\n%s",
			what, op, disassembleSPIRV(spirvBytes))
	}
}
