const std = @import("std");
pub const DataObject = struct {
    values: std.ArrayList(i32),
    grad: bool,
    attributes: std.StringHashMap(i32),

    pub fn init(allocator: std.mem.Allocator, grad: bool) DataObject {
        return .{
            .values = std.ArrayList(i32).init(allocator),
            .grad = grad,
        };
    }

    pub fn add(self: *DataObject, value: i32) !void {
        try self.values.append(value);
    }

    pub fn set(self: *DataObject, index: usize, value: i32) void {
        self.values.items[index] = value;
    }

    pub fn print(self: *DataObject) void {
        std.debug.print("{s}: {any}\n", .{ self.name, self.values.items });
    }
};
