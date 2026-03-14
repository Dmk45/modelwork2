const std = @import("std");
const trix = @import("matrix.zig");

// NN builder

pub fn builder(allocator: std.mem.Allocator, Size: usize){
const size = Size;
var data = try trix.DataObject.init(allocator, true);
for (0..Size) |i| {
    try data.add(i);
    try data.print();
}
return data;

}

main = builder(std.heap.page_allocator, 10);
