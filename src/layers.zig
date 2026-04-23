const std = @import("std");
const trix = @import("matrix.zig");
const core = @import("core_math.zig");
const grad_mod = @import("grad.zig");
const grad_math = @import("grad_math.zig");
const matlab_mod = @import("matlab.zig");

/// Layer configuration and state management
pub const LayerConfig = struct {
    input_size: usize,
    output_size: usize,
    activation: []const u8, // "relu", "sigmoid", "tanh", "none"
};

/// Linear layer state
pub const LinearLayer = struct {
    config: LayerConfig,
    weights: trix.DataObject,
    bias: trix.DataObject,

    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        activation: []const u8,
    ) !LinearLayer {
        var weights = try trix.DataObject.init(allocator, &[_]usize{ input_size, output_size }, .f32);
        var bias = try trix.DataObject.init(allocator, &[_]usize{output_size}, .f32);

        // Initialize with small random values
        const nano_ts = std.time.nanoTimestamp();
        const u64_ts: u64 = @intCast(@mod(nano_ts, std.math.maxInt(i64)));
        var prng = std.Random.DefaultPrng.init(u64_ts);
        var random = prng.random();

        for (weights.values.items) |*val| {
            val.* = (random.float(f32) - 0.5) * 0.1;
        }
        for (bias.values.items) |*val| {
            val.* = 0.0;
        }

        weights.enableGrad();
        bias.enableGrad();
        try weights.ensureGradValue();
        try bias.ensureGradValue();

        return LinearLayer{
            .config = LayerConfig{
                .input_size = input_size,
                .output_size = output_size,
                .activation = activation,
            },
            .weights = weights,
            .bias = bias,
        };
    }

    pub fn forward(self: *LinearLayer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        // Compute: output = input @ weights + bias
        var hidden = try core.matmul(allocator, input, &self.weights);
        hidden.enableGrad();
        try hidden.ensureGradValue();

        var output = try core.addBias(allocator, &hidden, &self.bias);
        output.enableGrad();
        try output.ensureGradValue();

        // Apply activation
        if (std.mem.eql(u8, self.config.activation, "relu")) {
            // Apply ReLU in-place - but we need a copy since we track values
            var activated = try trix.DataObject.init(allocator, output.shape.?.items, .f32);
            @memcpy(activated.values.items, output.values.items);
            matlab_mod.relu(activated.values.items.ptr, activated.values.items.len);
            activated.enableGrad();
            try activated.ensureGradValue();
            output.deinit();
            output = activated;
        } else if (std.mem.eql(u8, self.config.activation, "sigmoid")) {
            var activated = try trix.DataObject.init(allocator, output.shape.?.items, .f32);
            @memcpy(activated.values.items, output.values.items);
            matlab_mod.sigmoid(activated.values.items.ptr, activated.values.items.len);
            activated.enableGrad();
            try activated.ensureGradValue();
            output.deinit();
            output = activated;
        }

        hidden.deinit();
        return output;
    }

    pub fn deinit(self: *LinearLayer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *LinearLayer) void {
        if (self.weights.grad_value) |*g| {
            @memset(g.items, 0.0);
        }
        if (self.bias.grad_value) |*g| {
            @memset(g.items, 0.0);
        }
    }

    pub fn get_weights(self: *LinearLayer) *trix.DataObject {
        return &self.weights;
    }

    pub fn get_bias(self: *LinearLayer) *trix.DataObject {
        return &self.bias;
    }
};

/// Simple neural network container
pub const NeuralNetwork = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(LinearLayer),

    pub fn init(allocator: std.mem.Allocator) !NeuralNetwork {
        const layers = try std.ArrayList(LinearLayer).initCapacity(allocator, 10);
        return NeuralNetwork{
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn add_linear(
        self: *NeuralNetwork,
        input_size: usize,
        output_size: usize,
        activation: []const u8,
    ) !void {
        const layer = try LinearLayer.init(self.allocator, input_size, output_size, activation);
        try self.layers.append(self.allocator, layer);
    }

    pub fn forward(self: *NeuralNetwork, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var output = try trix.DataObject.init(allocator, input.shape.?.items, .f32);
        @memcpy(output.values.items, input.values.items);

        for (self.layers.items) |*layer| {
            const next_output = try layer.forward(allocator, &output);
            // Only deinit intermediate outputs, not the final one
            if (output.values.items.ptr != input.values.items.ptr) {
                output.deinit();
            }
            output = next_output;
        }

        return output;
    }

    pub fn zero_grad(self: *NeuralNetwork) void {
        for (self.layers.items) |*layer| {
            layer.zero_grad();
        }
    }

    pub fn update_parameters(self: *NeuralNetwork, optimizer: *grad_mod.Adam) !void {
        for (self.layers.items) |*layer| {
            if (layer.weights.grad_value) |_| {
                try optimizer.step(&layer.weights);
            }
            if (layer.bias.grad_value) |_| {
                try optimizer.step(&layer.bias);
            }
        }
    }

    pub fn deinit(self: *NeuralNetwork) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);
    }

    pub fn get_layer(self: *NeuralNetwork, idx: usize) ?*LinearLayer {
        if (idx < self.layers.items.len) {
            return &self.layers.items[idx];
        }
        return null;
    }

    pub fn num_layers(self: *NeuralNetwork) usize {
        return self.layers.items.len;
    }
};
