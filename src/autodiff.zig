const std = @import("std");
const trix = @import("matrix.zig");
const grad_math = @import("grad_math.zig");

pub const OpRecord = struct {
    op: grad_math.OperationType,
    input_ids: std.ArrayList(usize),
    output_id: usize,
    metadata: grad_math.OperationMetadata,
};

pub const Tape = struct {
    allocator: std.mem.Allocator,
    tensors: std.AutoHashMap(usize, *trix.DataObject),
    ops: std.ArrayList(OpRecord),
    next_id: usize,

    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{
            .allocator = allocator,
            .tensors = std.AutoHashMap(usize, *trix.DataObject).init(allocator),
            .ops = std.ArrayList(OpRecord).empty,
            .next_id = 1,
        };
    }

    pub fn deinit(self: *Tape) void {
        for (self.ops.items) |*op| {
            op.input_ids.deinit(self.allocator);
        }
        self.ops.deinit(self.allocator);
        self.tensors.deinit();
    }

    pub fn registerTensor(self: *Tape, tensor: *trix.DataObject) !usize {
        const id = self.next_id;
        self.next_id += 1;
        try self.tensors.put(id, tensor);
        return id;
    }

    pub fn record(
        self: *Tape,
        op: grad_math.OperationType,
        input_ids: []const usize,
        output_id: usize,
        metadata: grad_math.OperationMetadata,
    ) !void {
        var ids = try std.ArrayList(usize).initCapacity(self.allocator, input_ids.len);
        for (input_ids) |id| try ids.append(self.allocator, id);
        try self.ops.append(self.allocator, .{
            .op = op,
            .input_ids = ids,
            .output_id = output_id,
            .metadata = metadata,
        });
    }

    pub fn backward(self: *Tape) !void {
        var i = self.ops.items.len;
        while (i > 0) {
            i -= 1;
            const rec = self.ops.items[i];
            const out = self.tensors.get(rec.output_id) orelse continue;
            var input_buf = try self.allocator.alloc(*trix.DataObject, rec.input_ids.items.len);
            defer self.allocator.free(input_buf);
            for (rec.input_ids.items, 0..) |id, idx| {
                input_buf[idx] = self.tensors.get(id) orelse continue;
            }
            try grad_math.executeBackward(rec.op, input_buf, out, rec.metadata, self.allocator);
        }
    }
};

pub fn stopGradient(tensor: *trix.DataObject) void {
    tensor.grad = false;
    if (tensor.grad_value) |*g| {
        @memset(g.items, 0.0);
    }
}

pub fn detach(allocator: std.mem.Allocator, tensor: *trix.DataObject) !trix.DataObject {
    const out = try trix.DataObject.init(allocator, tensor.shape.?.items, tensor.dtype);
    @memcpy(out.values.items, tensor.values.items);
    return out;
}
