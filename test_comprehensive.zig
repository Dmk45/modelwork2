const std = @import("std");
const builder_mod = @import("maker.zig");
const grad_mod = @import("grad.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Test zeros for different dimensions
    std.debug.print("\n=== Testing Zeros ===\n", .{});

    var zeros_1d = try builder_mod.zeros(allocator, &[_]usize{ 4, 2 });
    defer zeros_1d.deinit();
    std.debug.print("1D Zeros: ", .{});
    zeros_1d.print();
    std.debug.print("\n", .{});

    // Test Mean Squared Error
    std.debug.print("\n=== Testing Mean Squared Error ===\n", .{});

    var y_pred = try builder_mod.zeros(allocator, &[_]usize{ 1, 4 });
    defer y_pred.deinit();
    y_pred.values.items[0] = 1.0;
    y_pred.values.items[1] = 2.0;
    y_pred.values.items[2] = 3.0;
    y_pred.values.items[3] = 4.0;
    y_pred.enableGrad();

    var y_true = try builder_mod.zeros(allocator, &[_]usize{ 1, 4 });
    defer y_true.deinit();
    y_true.values.items[0] = 1.5;
    y_true.values.items[1] = 2.5;
    y_true.values.items[2] = 3.5;
    y_true.values.items[3] = 4.5;

    const mse_loss = try grad_mod.meanSquaredError(&y_pred, &y_true);
    std.debug.print("MSE Loss: {d}\n", .{mse_loss});
    std.debug.print("Gradients: {any}\n", .{y_pred.grad_value.?.items});

    // Test Cross Entropy
    std.debug.print("\n=== Testing Cross Entropy ===\n", .{});

    var logits = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer logits.deinit();
    logits.values.items[0] = 1.0;
    logits.values.items[1] = 2.0;
    logits.values.items[2] = 0.5;
    logits.enableGrad();

    var targets = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer targets.deinit();
    targets.values.items[1] = 1.0; // one-hot for class 1

    const ce_loss = try grad_mod.crossEntropy(&logits, &targets, allocator);
    std.debug.print("Cross Entropy Loss: {d}\n", .{ce_loss});
    std.debug.print("Gradients: {any}\n", .{logits.grad_value.?.items});

    // Test Adam Optimizer
    std.debug.print("\n=== Testing Adam Optimizer ===\n", .{});

    var adam = try grad_mod.Adam.init(allocator, 0.01, 0.9, 0.999, 1e-8);
    defer adam.deinit();

    var param = try builder_mod.zeros(allocator, &[_]usize{ 1, 2 });
    defer param.deinit();
    param.values.items[0] = 1.0;
    param.values.items[1] = -1.0;
    param.enableGrad();

    // Simulate some gradients
    try param.ensureGradValue();
    param.grad_value.?.items[0] = 0.1;
    param.grad_value.?.items[1] = -0.2;

    std.debug.print("Before Adam step: {any}\n", .{param.values.items});
    try adam.step(&param);
    std.debug.print("After Adam step: {any}\n", .{param.values.items});
}
