const std = @import("std");
const trix = @import("matrix.zig");

// NN builder
pub fn builder(
    allocator: std.mem.Allocator,
    values: []const f32,
) !trix.DataObject {
    var data = try trix.DataObject.init(
        allocator,
        values.len,
        .f32,
    );

    for (values) |v| {
        try data.add(v);
    }

    return data;
}

pub fn zeros(
    allocator: std.mem.Allocator,
    size: usize,
) !trix.DataObject {
    var data = try trix.DataObject.init(
        allocator,
        size,
        .f32,
    );

    for (size) |i| {
        try data.add(0.0);
    }

    return data;
}

pub fn main() !void {
    var data = try builder(
        std.heap.page_allocator,
        &[_]f32{
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        },
    );

    defer data.deinit();

    std.debug.print("\nFinal object:\n", .{});
    data.print();
}
