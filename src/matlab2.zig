const std = @import("std");
const trix = @import("matrix.zig");

/// Element-wise addition: result = a + b
/// Both tensors must have the same shape
pub fn add(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    if (a.values.items.len != b.values.items.len) {
        return error.ShapeMismatch;
    }

    var result = try trix.DataObject.init(
        allocator,
        a.shape.?.items,
        .f32,
    );

    for (0..a.values.items.len) |i| {
        result.values.items[i] = a.values.items[i] + b.values.items[i];
    }

    return result;
}

/// Element-wise subtraction: result = a - b
/// Both tensors must have the same shape
pub fn sub(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    if (a.values.items.len != b.values.items.len) {
        return error.ShapeMismatch;
    }

    var result = try trix.DataObject.init(
        allocator,
        a.shape.?.items,
        .f32,
    );

    for (0..a.values.items.len) |i| {
        result.values.items[i] = a.values.items[i] - b.values.items[i];
    }

    return result;
}

/// Element-wise multiplication: result = a * b
/// Both tensors must have the same shape
pub fn mul(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    if (a.values.items.len != b.values.items.len) {
        return error.ShapeMismatch;
    }

    var result = try trix.DataObject.init(
        allocator,
        a.shape.?.items,
        .f32,
    );

    for (0..a.values.items.len) |i| {
        result.values.items[i] = a.values.items[i] * b.values.items[i];
    }

    return result;
}

/// Element-wise division: result = a / b
/// Both tensors must have the same shape
pub fn div(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    if (a.values.items.len != b.values.items.len) {
        return error.ShapeMismatch;
    }

    var result = try trix.DataObject.init(
        allocator,
        a.shape.?.items,
        .f32,
    );

    for (0..a.values.items.len) |i| {
        if (b.values.items[i] == 0.0) {
            return error.DivisionByZero;
        }
        result.values.items[i] = a.values.items[i] / b.values.items[i];
    }

    return result;
}

/// Scale tensor by scalar: result = tensor * scalar
pub fn scale(allocator: std.mem.Allocator, tensor: *trix.DataObject, scalar: f32) !trix.DataObject {
    var result = try trix.DataObject.init(
        allocator,
        tensor.shape.?.items,
        .f32,
    );

    for (0..tensor.values.items.len) |i| {
        result.values.items[i] = tensor.values.items[i] * scalar;
    }

    return result;
}

/// In-place scale: tensor *= scalar
pub fn scaleInPlace(tensor: *trix.DataObject, scalar: f32) void {
    for (0..tensor.values.items.len) |i| {
        tensor.values.items[i] *= scalar;
    }
}

/// Add scalar to all elements: result = tensor + scalar
pub fn addScalar(allocator: std.mem.Allocator, tensor: *trix.DataObject, scalar: f32) !trix.DataObject {
    var result = try trix.DataObject.init(
        allocator,
        tensor.shape.?.items,
        .f32,
    );

    for (0..tensor.values.items.len) |i| {
        result.values.items[i] = tensor.values.items[i] + scalar;
    }

    return result;
}

/// Calculate flat index from multi-dimensional coordinates using strides
fn flatIndex(coords: []const usize, strides: []const usize) usize {
    var idx: usize = 0;
    for (coords, 0..) |c, i| {
        if (i < strides.len) {
            idx += c * strides[i];
        }
    }
    return idx;
}

/// Matrix multiplication: result[i,j] = sum_k(a[i,k] * b[k,j])
/// Handles 2D matrices and batched 3D operations [batch, m, k] @ [batch, k, n] -> [batch, m, n]
pub fn matmul(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    const a_shape = a.shape.?.items;
    const b_shape = b.shape.?.items;

    // Determine if batched or not
    const is_batched = a_shape.len == 3 and b_shape.len == 3;

    if (is_batched) {
        // Batched matrix multiplication: [B, M, K] @ [B, K, N] -> [B, M, N]
        if (a_shape.len != 3 or b_shape.len != 3) {
            return error.ShapeMismatch;
        }

        const batch_size = a_shape[0];
        const m = a_shape[1];
        const k = a_shape[2];
        const n = b_shape[2];

        if (b_shape[0] != batch_size or b_shape[1] != k) {
            return error.ShapeMismatch;
        }

        var result = try trix.DataObject.init(
            allocator,
            &[_]usize{ batch_size, m, n },
            .f32,
        );

        for (0..batch_size) |b_idx| {
            for (0..m) |i| {
                for (0..n) |j| {
                    var acc: f32 = 0.0;
                    for (0..k) |ki| {
                        const a_idx = b_idx * m * k + i * k + ki;
                        const b_idx_flat = b_idx * k * n + ki * n + j;
                        acc += a.values.items[a_idx] * b.values.items[b_idx_flat];
                    }
                    try result.values.append(acc);
                }
            }
        }

        return result;
    } else {
        // Standard 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
        if (a_shape.len != 2 or b_shape.len != 2) {
            return error.ShapeMismatch;
        }

        const m = a_shape[0];
        const k = a_shape[1];
        const n = b_shape[1];

        if (b_shape[0] != k) {
            return error.ShapeMismatch;
        }

        var result = try trix.DataObject.init(
            allocator,
            &[_]usize{ m, n },
            .f32,
        );

        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0.0;
                for (0..k) |ki| {
                    acc += a.values.items[i * k + ki] * b.values.items[ki * n + j];
                }
                result.values.items[i * n + j] = acc;
            }
        }

        return result;
    }
}

/// Transpose last two dimensions of a matrix
/// For 2D [M, N] -> [N, M]
/// For 3D [B, M, N] -> [B, N, M]
pub fn transpose(allocator: std.mem.Allocator, tensor: *trix.DataObject) !trix.DataObject {
    const shape = tensor.shape.?.items;

    if (shape.len == 2) {
        // 2D transpose
        const m = shape[0];
        const n = shape[1];

        var result = try trix.DataObject.init(allocator, &[_]usize{ n, m }, .f32);

        for (0..m) |i| {
            for (0..n) |j| {
                result.values.items[j * m + i] = tensor.values.items[i * n + j];
            }
        }

        return result;
    } else if (shape.len == 3) {
        // 3D transpose (swap last two dims)
        const b = shape[0];
        const m = shape[1];
        const n = shape[2];

        var result = try trix.DataObject.init(allocator, &[_]usize{ b, n, m }, .f32);

        for (0..b) |batch| {
            for (0..m) |i| {
                for (0..n) |j| {
                    const src_idx = batch * m * n + i * n + j;
                    const dst_idx = batch * n * m + j * m + i;
                    result.values.items[dst_idx] = tensor.values.items[src_idx];
                }
            }
        }

        return result;
    } else {
        return error.UnsupportedRank;
    }
}

/// Reshape tensor to new shape (no copy, just metadata change)
/// Total size must match
pub fn reshape(tensor: *trix.DataObject, new_shape: []const usize, allocator: std.mem.Allocator) !void {
    // Calculate total size
    var old_size: usize = 1;
    for (tensor.shape.?.items) |dim| {
        old_size *= dim;
    }

    var new_size: usize = 1;
    for (new_shape) |dim| {
        new_size *= dim;
    }

    if (old_size != new_size) {
        return error.ReshapeMismatch;
    }

    // Update shape
    tensor.shape.?.clearAndFree();
    for (new_shape) |dim| {
        try tensor.shape.?.append(dim);
    }

    // Recalculate strides
    tensor.strides.?.clearAndFree();
    var total_size: usize = 1;
    var i: usize = new_shape.len;
    while (i > 0) {
        i -= 1;
        try tensor.strides.?.insert(allocator, 0, total_size);
        total_size *= new_shape[i];
    }
}

/// Sum all elements in tensor
pub fn sum(tensor: *trix.DataObject) f32 {
    var result: f32 = 0.0;
    for (tensor.values.items) |val| {
        result += val;
    }
    return result;
}

/// Mean of all elements in tensor
pub fn mean(tensor: *trix.DataObject) f32 {
    const total = sum(tensor);
    return total / @as(f32, @floatFromInt(tensor.values.items.len));
}

/// Dot product of two 1D tensors
pub fn dot(a: *trix.DataObject, b: *trix.DataObject) !f32 {
    if (a.values.items.len != b.values.items.len) {
        return error.ShapeMismatch;
    }

    var acc: f32 = 0.0;
    for (0..a.values.items.len) |i| {
        acc += a.values.items[i] * b.values.items[i];
    }
    return acc;
}

/// Add bias to each row of a 2D matrix or batch
/// For 2D [M, N] + bias[N]: adds bias[j] to each row i
/// For 3D [B, M, N] + bias[N]: adds bias[j] to each batch and row
pub fn addBias(allocator: std.mem.Allocator, matrix: *trix.DataObject, bias: *trix.DataObject) !trix.DataObject {
    const shape = matrix.shape.?.items;
    const bias_shape = bias.shape.?.items;

    if (shape.len == 2) {
        // 2D case: [M, N] + [N]
        const m = shape[0];
        const n = shape[1];

        if (bias_shape.len != 1 or bias_shape[0] != n) {
            return error.ShapeMismatch;
        }

        var result = try trix.DataObject.init(allocator, shape, .f32);

        for (0..m) |i| {
            for (0..n) |j| {
                result.values.items[i * n + j] = matrix.values.items[i * n + j] + bias.values.items[j];
            }
        }

        return result;
    } else if (shape.len == 3) {
        // 3D case: [B, M, N] + [N]
        const b = shape[0];
        const m = shape[1];
        const n = shape[2];

        if (bias_shape.len != 1 or bias_shape[0] != n) {
            return error.ShapeMismatch;
        }

        var result = try trix.DataObject.init(allocator, shape, .f32);

        for (0..b) |batch| {
            for (0..m) |i| {
                for (0..n) |j| {
                    const idx = batch * m * n + i * n + j;
                    result.values.items[idx] = matrix.values.items[idx] + bias.values.items[j];
                }
            }
        }

        return result;
    } else {
        return error.UnsupportedRank;
    }
}

/// Matrix-Vector multiplication: [M, N] @ [N] -> [M]
pub fn matvec(allocator: std.mem.Allocator, matrix: *trix.DataObject, vec: *trix.DataObject) !trix.DataObject {
    const m_shape = matrix.shape.?.items;
    const v_shape = vec.shape.?.items;

    if (m_shape.len != 2 or v_shape.len != 1) {
        return error.ShapeMismatch;
    }

    const m = m_shape[0];
    const n = m_shape[1];

    if (v_shape[0] != n) {
        return error.ShapeMismatch;
    }

    var result = try trix.DataObject.init(allocator, &[_]usize{m}, .f32);

    for (0..m) |i| {
        var acc: f32 = 0.0;
        for (0..n) |j| {
            acc += matrix.values.items[i * n + j] * vec.values.items[j];
        }
        result.values.items[i] = acc;
    }

    return result;
}

/// Outer product of two 1D vectors: [M] ⊗ [N] -> [M, N]
pub fn outer(allocator: std.mem.Allocator, a: *trix.DataObject, b: *trix.DataObject) !trix.DataObject {
    const a_shape = a.shape.?.items;
    const b_shape = b.shape.?.items;

    if (a_shape.len != 1 or b_shape.len != 1) {
        return error.ShapeMismatch;
    }

    const m = a_shape[0];
    const n = b_shape[0];

    var result = try trix.DataObject.init(allocator, &[_]usize{ m, n }, .f32);

    for (0..m) |i| {
        for (0..n) |j| {
            result.values.items[i * n + j] = a.values.items[i] * b.values.items[j];
        }
    }

    return result;
}

/// Frobenius norm: sqrt(sum of all squared elements)
pub fn frobeniusNorm(tensor: *trix.DataObject) f32 {
    var sum_sq: f32 = 0.0;
    for (tensor.values.items) |val| {
        sum_sq += val * val;
    }
    return std.math.sqrt(sum_sq);
}

/// Clamp all values to [min, max]
pub fn clamp(tensor: *trix.DataObject, min_val: f32, max_val: f32) void {
    for (0..tensor.values.items.len) |i| {
        if (tensor.values.items[i] < min_val) {
            tensor.values.items[i] = min_val;
        } else if (tensor.values.items[i] > max_val) {
            tensor.values.items[i] = max_val;
        }
    }
}

/// Element-wise absolute value
pub fn abs(allocator: std.mem.Allocator, tensor: *trix.DataObject) !trix.DataObject {
    var result = try trix.DataObject.init(allocator, tensor.shape.?.items, .f32);

    for (0..tensor.values.items.len) |i| {
        result.values.items[i] = std.math.fabs(tensor.values.items[i]);
    }

    return result;
}
