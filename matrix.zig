const std = @import("std");
pub fn brew() !void {
    std.debug.print("Hello, world!\n", .{});
}

pub fn main() !void {
    brew();
}
