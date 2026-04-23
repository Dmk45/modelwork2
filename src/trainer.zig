const std = @import("std");
const trix = @import("matrix.zig");
const layers = @import("layers.zig");
const grad = @import("grad.zig");

pub const EpochStats = struct {
    train_loss: f32,
    val_loss: f32,
};

pub const TrainingHistory = struct {
    allocator: std.mem.Allocator,
    epochs: std.ArrayList(EpochStats),

    pub fn init(allocator: std.mem.Allocator) TrainingHistory {
        return .{ .allocator = allocator, .epochs = std.ArrayList(EpochStats).empty };
    }

    pub fn deinit(self: *TrainingHistory) void {
        self.epochs.deinit(self.allocator);
    }
};

pub const TrainerConfig = struct {
    epochs: usize = 1,
    clip_grad_norm: ?f32 = null,
};

pub const Trainer = struct {
    allocator: std.mem.Allocator,
    model: *layers.NeuralNetwork,
    optimizer: *grad.Adam,
    config: TrainerConfig,
    history: TrainingHistory,
    best_val_loss: f32,

    pub fn init(allocator: std.mem.Allocator, model: *layers.NeuralNetwork, optimizer: *grad.Adam, config: TrainerConfig) Trainer {
        return .{
            .allocator = allocator,
            .model = model,
            .optimizer = optimizer,
            .config = config,
            .history = TrainingHistory.init(allocator),
            .best_val_loss = std.math.inf(f32),
        };
    }

    pub fn deinit(self: *Trainer) void {
        self.history.deinit();
    }

    pub fn trainEpoch(self: *Trainer, x: *trix.DataObject, y: *trix.DataObject) !f32 {
        var pred = try self.model.forward(self.allocator, x);
        defer pred.deinit();
        const loss = try grad.meanSquaredError(&pred, y);
        if (self.config.clip_grad_norm) |max_norm| {
            for (self.model.layers.items) |*layer| {
                grad.clipGradientsByNorm(&layer.weights, max_norm);
                grad.clipGradientsByNorm(&layer.bias, max_norm);
            }
        }
        try self.model.update_parameters(self.optimizer);
        return loss;
    }

    pub fn evaluate(self: *Trainer, x: *trix.DataObject, y: *trix.DataObject) !f32 {
        var pred = try self.model.forward(self.allocator, x);
        defer pred.deinit();
        return grad.meanSquaredError(&pred, y);
    }

    pub fn fit(self: *Trainer, train_x: *trix.DataObject, train_y: *trix.DataObject, val_x: *trix.DataObject, val_y: *trix.DataObject) !void {
        for (0..self.config.epochs) |_| {
            const train_loss = try self.trainEpoch(train_x, train_y);
            const val_loss = try self.evaluate(val_x, val_y);
            if (val_loss < self.best_val_loss) self.best_val_loss = val_loss;
            try self.history.epochs.append(self.allocator, .{ .train_loss = train_loss, .val_loss = val_loss });
        }
    }

    pub fn saveCheckpoint(self: *Trainer, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        const payload = try std.fmt.allocPrint(self.allocator, "best_val_loss={d}\nnum_layers={}\n", .{ self.best_val_loss, self.model.layers.items.len });
        defer self.allocator.free(payload);
        try file.writeAll(payload);
    }
};
