const std = @import("std");
const trix = @import("matrix.zig");

fn getRank(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .Array => |info| 1 + getRank(info.child),
        else => 0,
    };
}

fn getShape(comptime T: type) [getRank(T)]usize {
    const rank = comptime getRank(T);
    var res: [rank]usize = undefined;
    var CurrentType = T;
    inline for (0..rank) |i| {
        const info = @typeInfo(CurrentType).Array;
        res[i] = info.len;
        CurrentType = info.child;
    }
    return res;
}

pub fn builder(
    allocator: std.mem.Allocator,
    values: anytype,
) !trix.DataObject {
    const T = @TypeOf(values);
    const shape = comptime getShape(T);

    const data = try trix.DataObject.init(allocator, &shape, .f32);

    // Flatten nested arrays into the allocated slice
    const flat_ptr: [*]const f32 = @ptrCast(&values);
    @memcpy(data.values, flat_ptr[0..data.values.len]);

    return data;
}
pub fn zeros(allocator: std.mem.Allocator, shape: []const usize) !trix.DataObject {
    const data = try trix.DataObject.init(
        allocator,
        shape,
        .f32,
    );

    @memset(data.values, 0.0);

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
