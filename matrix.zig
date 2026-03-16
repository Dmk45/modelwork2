const std = @import("std");

pub const DType = enum {
    f32,
    f64,
    i32,
    i64,
};

pub const DataObject = struct {
    allocator: std.mem.Allocator,

    // tensor data
    values: std.ArrayList(f32),

    // tensor metadata
    shape: ?std.ArrayList(usize),
    dtype: DType,

    // autograd
    grad: ?bool,
    grad_value: ?std.ArrayList(f32),

    // misc metadata
    attributes: ?std.StringHashMap(f32),

    pub fn init(
        allocator: std.mem.Allocator,
        capacity: usize,
        dtype: DType,
    ) !DataObject {
        return .{
            .allocator = allocator,
            .values = try std.ArrayList(f32).initCapacity(allocator, capacity),

            .shape = null,
            .dtype = dtype,

            .grad = null,
            .grad_value = null,

            .attributes = null,
        };
    }

    pub fn remove(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.orderedRemove(index);
    }

    // Fast remove (does not preserve order)
    pub fn swapRemove(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.swapRemove(index);
    }

    pub fn deinit(self: *DataObject) void {
        self.values.deinit(self.allocator);

        if (self.shape) |*s| {
            s.deinit(self.allocator);
        }

        if (self.grad_value) |*g| {
            g.deinit(self.allocator);
        }

        if (self.attributes) |*a| {
            a.deinit();
        }
    }

    pub fn add(self: *DataObject, value: f32) !void {
        try self.values.append(self.allocator, value);
    }

    pub fn get(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.items[index];
    }

    fn ensureShape(self: *DataObject) !void {
        if (self.shape == null) {
            self.shape = try std.ArrayList(usize).initCapacity(self.allocator, 4);
        }
    }

    pub fn addDim(self: *DataObject, dim: usize) !void {
        try self.ensureShape();
        try self.shape.?.append(self.allocator, dim);
    }

    pub fn enableGrad(self: *DataObject) void {
        self.grad = true;
    }

    fn ensureGradValue(self: *DataObject) !void {
        if (self.grad_value == null) {
            self.grad_value = try std.ArrayList(f32).initCapacity(self.allocator, self.values.items.len);
        }
    }

    pub fn addGrad(self: *DataObject, value: f32) !void {
        try self.ensureGradValue();
        try self.grad_value.?.append(self.allocator, value);
    }

    fn ensureAttributes(self: *DataObject) void {
        if (self.attributes == null) {
            self.attributes = std.StringHashMap(f32).init(self.allocator);
        }
    }

    pub fn setAttr(self: *DataObject, key: []const u8, value: f32) !void {
        self.ensureAttributes();
        try self.attributes.?.put(key, value);
    }

    pub fn getAttr(self: *DataObject, key: []const u8) ?f32 {
        if (self.attributes == null)
            return null;

        return self.attributes.?.get(key);
    }

    pub fn print(self: *const DataObject) void {
        std.debug.print("DataObject\n", .{});

        std.debug.print("dtype: {any}\n", .{self.dtype});

        std.debug.print("values: {any}\n", .{self.values.items});

        if (self.shape) |s| {
            std.debug.print("shape: {any}\n", .{s.items});
        } else {
            std.debug.print("shape: <none>\n", .{});
        }

        if (self.grad) |g| {
            std.debug.print("grad enabled: {}\n", .{g});
        } else {
            std.debug.print("grad enabled: <none>\n", .{});
        }

        if (self.grad_value) |g| {
            std.debug.print("grad values: {any}\n", .{g.items});
        } else {
            std.debug.print("grad values: <none>\n", .{});
        }

        if (self.attributes) |attrs| {
            var it = attrs.iterator();
            while (it.next()) |entry| {
                std.debug.print("attr {s}: {d}\n", .{
                    entry.key_ptr.*,
                    entry.value_ptr.*,
                });
            }
        }
    }
};
