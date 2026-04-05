const std = @import("std");
const builder_mod = @import("maker.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Test zeros for different dimensions
    std.debug.print("\n=== Testing Zeros ===\n", .{});

    var zeros_1d = try builder_mod.zeros(allocator, &[_]usize{ 4, 2 });
    defer zeros_1d.deinit();
    std.debug.print("1D Zeros: ", .{});
    zeros_1d.print();
    std.debug.print("\n", .{});
}
