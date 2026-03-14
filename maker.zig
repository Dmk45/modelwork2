const std = @import("std");
const trix = @import("matrix.zig");

// NN builder

pub fn builder(allocator: std.mem.Allocator, Size: usize) !trix.DataObject {
    const size = Size;
    var data = try trix.DataObject.init(allocator, true);
    for (0..size) |i| {
        try data.add(i);
        try data.print();
    }

    return data;
}

pub fn main() !void {
    // Call the builder inside a function (like main)
    const data = try builder(std.heap.page_allocator, 10);
    return data;
    // Remember to deinit if your DataObject requires it
    // defer data.deinit();
}
