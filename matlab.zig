const std = @import("std");

pub fn relu(buffer: [*]f32, size: usize) void {
    for (0..size) |i| {
        if (buffer[i] < 0.0) {
            buffer[i] = 0.0;
        }
    }
}

pub fn sigmoid(buffer: [*]f32, size: usize) void {
    for (0..size) |i| {
        buffer[i] = 1.0 / (1.0 + std.math.exp(-buffer[i]));
    }
}

pub fn softmax(buffer: [*]f32, size: usize) void {
    // Find the maximum value for numerical stability
    var max_val: f32 = -std.math.inf(f32);
    for (0..size) |i| {
        if (buffer[i] > max_val) {
            max_val = buffer[i];
        }
    }

    // Compute exp(x - max) and accumulate sum
    var sum: f32 = 0.0;
    for (0..size) |i| {
        buffer[i] = std.math.exp(buffer[i] - max_val);
        sum += buffer[i];
    }

    // Normalize by dividing by the sum
    for (0..size) |i| {
        buffer[i] /= sum;
    }
}
