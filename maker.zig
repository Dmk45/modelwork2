const std = @import("std");
const trix = @import("matrix.zig");
pub fn builder(
    allocator: std.mem.Allocator,
    values: []const f32,
) !trix.DataObject {
    var data = try trix.DataObject.init(
        allocator,
        values.len,
        .f32,
    );

    for (values) |value| {
        try data.add(value);
    }

    return data;
}

pub fn zeros(
    allocator: std.mem.Allocator,
    shape: std.AutoHashMap([]const u8, usize),
) !trix.DataObject {
    // 1. Get dimensions safely using optional unwrapping
    const rows = shape.get("rows") orelse return error.MissingRows;
    const cols = shape.get("cols") orelse return error.MissingCols;
    const total_size = rows * cols;

    // 2. Initialize the data object with the calculated total size
    var data = try trix.DataObject.init(
        allocator,
        total_size,
        .f32,
    );

    // 3. Fill with zeros
    // If DataObject h mas an 'add' method:
    for (0..total_size) |_| {
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

// TODO: Able to change Matrix size and type by modifying the builder function and the zeros function.
