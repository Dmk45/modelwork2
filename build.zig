const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const framework_module = b.addModule("modelwork2", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const comprehensive_example = b.addExecutable(.{
        .name = "example-comprehensive",
        .root_source_file = b.path("examples/test_comprehensive.zig"),
        .target = target,
        .optimize = optimize,
    });
    comprehensive_example.root_module.addImport("modelwork2", framework_module);
    b.installArtifact(comprehensive_example);

    const layers_example = b.addExecutable(.{
        .name = "example-layers",
        .root_source_file = b.path("examples/test_layers.zig"),
        .target = target,
        .optimize = optimize,
    });
    layers_example.root_module.addImport("modelwork2", framework_module);
    b.installArtifact(layers_example);

    const run_comprehensive = b.addRunArtifact(comprehensive_example);
    const run_layers = b.addRunArtifact(layers_example);

    const run_step = b.step("run", "Run all examples");
    run_step.dependOn(&run_comprehensive.step);
    run_step.dependOn(&run_layers.step);

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("tests/framework_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    unit_tests.root_module.addImport("modelwork2", framework_module);
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Generate docs using Zig's emit-docs flow.
    const docs_cmd = b.addSystemCommand(&[_][]const u8{
        "zig",
        "test",
        "src/lib.zig",
        "-femit-docs",
        "--name",
        "modelwork2-docs",
    });
    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&docs_cmd.step);
}
