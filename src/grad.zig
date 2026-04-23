const std = @import("std");
const trix = @import("matrix.zig");

/// Computes the softmax of the input tensor along the last dimension
pub fn softmax(allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    // Assume input is 2D: batch_size x num_classes
    const batch_size = input.shape.?.items[0];
    const num_classes = input.shape.?.items[1];

    var output = try trix.DataObject.init(allocator, &[_]usize{ batch_size, num_classes }, .f32);

    for (0..batch_size) |b| {
        // Find max for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        for (0..num_classes) |c| {
            const val = input.get(&[_]usize{ b, c });
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        var sum_exp: f32 = 0.0;
        for (0..num_classes) |c| {
            const val = input.get(&[_]usize{ b, c });
            const exp_val = std.math.exp(val - max_val);
            try output.set(b * num_classes + c, exp_val);
            sum_exp += exp_val;
        }

        // Normalize
        for (0..num_classes) |c| {
            const idx = b * num_classes + c;
            const val = output.values.items[idx];
            output.values.items[idx] = val / sum_exp;
        }
    }

    return output;
}

/// Mean Squared Error loss
pub fn meanSquaredError(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) {
        return error.ShapeMismatch;
    }

    var loss: f32 = 0.0;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));

    for (0..y_pred.values.items.len) |i| {
        const diff = y_pred.values.items[i] - y_true.values.items[i];
        loss += diff * diff;
    }

    loss /= n;

    // Compute gradients if enabled
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const diff = y_pred.values.items[i] - y_true.values.items[i];
            y_pred.grad_value.?.items[i] += (2.0 * diff) / n;
        }
    }

    return loss;
}

/// Cross Entropy loss (assumes y_pred is logits, y_true is one-hot encoded)
pub fn crossEntropy(y_pred: *trix.DataObject, y_true: *trix.DataObject, allocator: std.mem.Allocator) !f32 {
    // Apply softmax to y_pred
    var probs = try softmax(allocator, y_pred);
    defer probs.deinit();

    var loss: f32 = 0.0;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));

    for (0..y_pred.values.items.len) |i| {
        if (y_true.values.items[i] > 0.0) {
            loss -= y_true.values.items[i] * std.math.log(f32, std.math.e, probs.values.items[i] + 1e-7);
        }
    }

    loss /= n;

    // Compute gradients if enabled
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            y_pred.grad_value.?.items[i] += (probs.values.items[i] - y_true.values.items[i]) / n;
        }
    }

    return loss;
}

/// Adam Optimizer
pub const Adam = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: ?std.ArrayList(f32),
    v: ?std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, beta1: f32, beta2: f32, epsilon: f32) !Adam {
        return Adam{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .t = 0,
            .m = null,
            .v = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Adam) void {
        if (self.m) |*m| m.deinit(self.allocator);
        if (self.v) |*v| v.deinit(self.allocator);
    }

    pub fn step(self: *Adam, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        self.t += 1;

        // Initialize m and v if not done
        if (self.m == null) {
            self.m = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.m.?.resize(self.allocator, param.values.items.len);
            @memset(self.m.?.items, 0.0);
        }
        if (self.v == null) {
            self.v = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.v.?.resize(self.allocator, param.values.items.len);
            @memset(self.v.?.items, 0.0);
        }

        const t_f = @as(f32, @floatFromInt(self.t));

        for (0..param.values.items.len) |i| {
            const g = param.grad_value.?.items[i];

            // Update biased first moment estimate
            self.m.?.items[i] = self.beta1 * self.m.?.items[i] + (1.0 - self.beta1) * g;

            // Update biased second raw moment estimate
            self.v.?.items[i] = self.beta2 * self.v.?.items[i] + (1.0 - self.beta2) * g * g;

            // Compute bias-corrected first moment estimate
            const m_hat = self.m.?.items[i] / (1.0 - std.math.pow(f32, self.beta1, t_f));

            // Compute bias-corrected second raw moment estimate
            const v_hat = self.v.?.items[i] / (1.0 - std.math.pow(f32, self.beta2, t_f));

            // Update parameter
            param.values.items[i] -= self.lr * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
        }

        // Zero gradients after update
        @memset(param.grad_value.?.items, 0.0);
    }
};
