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
        if (self.m) |*m| m.deinit();
        if (self.v) |*v| v.deinit();
    }

    pub fn step(self: *Adam, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        self.t += 1;

        // Initialize m and v if not done
        if (self.m == null) {
            self.m = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.m.?.resize(param.values.items.len);
            @memset(self.m.?.items, 0.0);
        }
        if (self.v == null) {
            self.v = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.v.?.resize(param.values.items.len);
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

/// SGD optimizer with optional momentum and Nesterov acceleration.
pub const SGD = struct {
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    velocity: ?std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, momentum: f32, nesterov: bool, weight_decay: f32) SGD {
        return .{
            .lr = lr,
            .momentum = momentum,
            .nesterov = nesterov,
            .weight_decay = weight_decay,
            .velocity = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SGD) void {
        if (self.velocity) |*v| v.deinit();
    }

    pub fn step(self: *SGD, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.velocity == null) {
            self.velocity = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.velocity.?.resize(param.values.items.len);
            @memset(self.velocity.?.items, 0.0);
        }

        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.velocity.?.items[i] = self.momentum * self.velocity.?.items[i] + grad;
            const update = if (self.nesterov) grad + self.momentum * self.velocity.?.items[i] else self.velocity.?.items[i];
            param.values.items[i] -= self.lr * update;
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

/// RMSprop optimizer.
pub const RMSprop = struct {
    lr: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    square_avg: ?std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, alpha: f32, epsilon: f32, weight_decay: f32) RMSprop {
        return .{
            .lr = lr,
            .alpha = alpha,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .square_avg = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RMSprop) void {
        if (self.square_avg) |*s| s.deinit();
    }

    pub fn step(self: *RMSprop, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.square_avg == null) {
            self.square_avg = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.square_avg.?.resize(param.values.items.len);
            @memset(self.square_avg.?.items, 0.0);
        }
        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.square_avg.?.items[i] = self.alpha * self.square_avg.?.items[i] + (1.0 - self.alpha) * grad * grad;
            param.values.items[i] -= self.lr * grad / (std.math.sqrt(self.square_avg.?.items[i]) + self.epsilon);
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

/// AdaGrad optimizer.
pub const AdaGrad = struct {
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    sum_sq: ?std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, epsilon: f32, weight_decay: f32) AdaGrad {
        return .{
            .lr = lr,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .sum_sq = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdaGrad) void {
        if (self.sum_sq) |*s| s.deinit();
    }

    pub fn step(self: *AdaGrad, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.sum_sq == null) {
            self.sum_sq = try std.ArrayList(f32).initCapacity(self.allocator, param.values.items.len);
            try self.sum_sq.?.resize(param.values.items.len);
            @memset(self.sum_sq.?.items, 0.0);
        }
        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.sum_sq.?.items[i] += grad * grad;
            param.values.items[i] -= self.lr * grad / (std.math.sqrt(self.sum_sq.?.items[i]) + self.epsilon);
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

pub const StepLR = struct {
    step_size: usize,
    gamma: f32,
    pub fn step(self: StepLR, base_lr: f32, epoch: usize) f32 {
        const k = @divFloor(epoch, self.step_size);
        return base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(k)));
    }
};

pub const ExponentialLR = struct {
    gamma: f32,
    pub fn step(self: ExponentialLR, base_lr: f32, epoch: usize) f32 {
        return base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(epoch)));
    }
};

pub const WarmupLR = struct {
    warmup_epochs: usize,
    target_lr: f32,
    pub fn step(self: WarmupLR, epoch: usize) f32 {
        if (self.warmup_epochs == 0 or epoch >= self.warmup_epochs) return self.target_lr;
        const t = @as(f32, @floatFromInt(epoch + 1)) / @as(f32, @floatFromInt(self.warmup_epochs));
        return self.target_lr * t;
    }
};

pub fn binaryCrossEntropy(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
        const y = y_true.values.items[i];
        loss += -(y * std.math.log(f32, std.math.e, p) + (1.0 - y) * std.math.log(f32, std.math.e, 1.0 - p));
    }
    loss /= n;
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
            const y = y_true.values.items[i];
            y_pred.grad_value.?.items[i] += (p - y) / (p * (1.0 - p) * n);
        }
    }
    return loss;
}

pub fn l1Loss(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        loss += @abs(y_pred.values.items[i] - y_true.values.items[i]);
    }
    return loss / n;
}

pub fn smoothL1Loss(y_pred: *trix.DataObject, y_true: *trix.DataObject, beta: f32) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    if (beta <= 0.0) return error.InvalidBeta;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const d = @abs(y_pred.values.items[i] - y_true.values.items[i]);
        if (d < beta) {
            loss += 0.5 * d * d / beta;
        } else {
            loss += d - 0.5 * beta;
        }
    }
    return loss / n;
}

pub fn hingeLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const margin = 1.0 - y_true.values.items[i] * y_pred.values.items[i];
        if (margin > 0.0) loss += margin;
    }
    return loss / n;
}

pub fn clipGradientsByValue(param: *trix.DataObject, min_val: f32, max_val: f32) void {
    if (param.grad_value == null) return;
    for (param.grad_value.?.items) |*g| {
        g.* = std.math.clamp(g.*, min_val, max_val);
    }
}

pub fn clipGradientsByNorm(param: *trix.DataObject, max_norm: f32) void {
    if (param.grad_value == null or max_norm <= 0.0) return;
    var norm: f32 = 0.0;
    for (param.grad_value.?.items) |g| norm += g * g;
    norm = std.math.sqrt(norm);
    if (norm <= max_norm) return;
    const scale = max_norm / (norm + 1e-12);
    for (param.grad_value.?.items) |*g| g.* *= scale;
}

/// Accumulate external gradients into a parameter gradient buffer.
pub fn accumulateGradients(param: *trix.DataObject, grads: []const f32, average_by: usize) !void {
    if (grads.len != param.values.items.len) return error.ShapeMismatch;
    try param.ensureGradValue();
    const denom = if (average_by == 0) 1.0 else @as(f32, @floatFromInt(average_by));
    for (grads, 0..) |g, i| {
        param.grad_value.?.items[i] += g / denom;
    }
}
