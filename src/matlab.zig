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

pub fn tanh(buffer: [*]f32, size: usize) void {
    for (0..size) |i| {
        buffer[i] = std.math.tanh(buffer[i]);
    }
}

pub fn elu(buffer: [*]f32, size: usize, alpha: f32) void {
    for (0..size) |i| {
        const x = buffer[i];
        buffer[i] = if (x > 0.0) x else alpha * (std.math.exp(x) - 1.0);
    }
}

pub fn leakyRelu(buffer: [*]f32, size: usize, negative_slope: f32) void {
    for (0..size) |i| {
        const x = buffer[i];
        buffer[i] = if (x > 0.0) x else negative_slope * x;
    }
}

pub fn gelu(buffer: [*]f32, size: usize) void {
    for (0..size) |i| {
        const x = buffer[i];
        // GELU approximation: x * Φ(x) where Φ(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const sqrt_2_pi = std.math.sqrt(2.0 / std.math.pi);
        const inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        const cdf = 0.5 * (1.0 + std.math.tanh(inner));
        buffer[i] = x * cdf;
    }
}
