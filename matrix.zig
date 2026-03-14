const std = @import("std");

pub const DataObject = struct {
    values: std.ArrayList(f32),
    grad: bool,
    attributes: std.StringHashMap(f32),

    pub fn init(allocator: std.mem.Allocator, grad: bool) !DataObject {
        return .{
            .grad = grad,
            .values = try std.ArrayList(f32).initCapacity(allocator, 0),
            .attributes = std.StringHashMap(f32).init(allocator),
        };
    }

    pub fn deinit(self: *DataObject) void {
        self.values.deinit();
        self.attributes.deinit();
    }

    pub fn add(self: *DataObject, value: f32) !void {
        try self.values.append(value);
    }

    pub fn set(self: *DataObject, index: usize, value: f32) !void {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        self.values.items[index] = value;
    }

    pub fn get(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.items[index];
    }

    pub fn setAttr(self: *DataObject, key: []const u8, value: f32) !void {
        try self.attributes.put(key, value);
    }

    pub fn getAttr(self: *DataObject, key: []const u8) ?f32 {
        return self.attributes.get(key);
    }

    pub fn print(self: *DataObject) void {
        std.debug.print("DataObject\n", .{});
        std.debug.print("grad: {}\n", .{self.grad});
        std.debug.print("values: {any}\n", .{self.values.items});

        var it = self.attributes.iterator();
        while (it.next()) |entry| {
            std.debug.print("attr {s}: {d}\n", .{
                entry.key_ptr.*,
                entry.value_ptr.*,
            });
        }
    }
};
