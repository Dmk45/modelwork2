const std = @import("std");
const trix = @import("matrix.zig");

pub fn builder(
    allocator: std.mem.Allocator,
    values: anytype,
) !trix.DataObject {
    const T = @TypeOf(values);
    const shape = comptime trix.shape(T);

    const data = try trix.DataObject.init(allocator, &shape, .f32);

    // Flatten nested arrays into the allocated slice
    const flat_ptr: [*]const f32 = @ptrCast(&values);
    @memcpy(data.values.items, flat_ptr[0..data.values.items.len]);

    return data;
}
pub fn zeros(allocator: std.mem.Allocator, shape: []const usize) !trix.DataObject {
    const data = try trix.DataObject.init(
        allocator,
        shape,
        .f32,
    );

    @memset(data.values.items, 0.0);

    return data;
}
