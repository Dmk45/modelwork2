const std = @import("std");
const lib = @import("modelwork2");

test "core modules compile and basic tensor operations run" {
    const allocator = std.testing.allocator;

    var a = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer a.deinit();

    var b = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer b.deinit();
    b.values.items[0] = 1.0;
    b.values.items[1] = 1.0;
    b.values.items[2] = 1.0;

    var c = try lib.core_math.add(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(@as(usize, 3), c.values.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.values.items[0], 1e-6);
}

test "data pipeline loads csv and exposes rows" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const csv_content =
        \\f1,f2,label
        \\1.0,2.0,0.0
        \\3.0,4.0,1.0
    ;
    try tmp.dir.writeFile(.{ .sub_path = "sample.csv", .data = csv_content });

    const cwd = std.fs.cwd();
    var prev_dir = try cwd.openDir(".", .{});
    defer prev_dir.close();
    try tmp.dir.setAsCwd();
    defer prev_dir.setAsCwd() catch {};

    var dataset = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, "sample.csv", true);
    defer dataset.deinit();

    try std.testing.expectEqual(@as(usize, 2), dataset.row_count);
    try std.testing.expectEqual(@as(usize, 2), dataset.feature_count);

    const row0 = try dataset.rowFeatures(0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), row0[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try dataset.rowLabel(0), 1e-6);
}

test "data pipeline split and batching are deterministic" {
    const allocator = std.testing.allocator;

    var split = try lib.data_pipeline.splitIndices(allocator, 10, 0.6, 0.2, 42);
    defer split.deinit();

    try std.testing.expectEqual(@as(usize, 6), split.train_indices.items.len);
    try std.testing.expectEqual(@as(usize, 2), split.val_indices.items.len);
    try std.testing.expectEqual(@as(usize, 2), split.test_indices.items.len);

    var iter = try lib.data_pipeline.BatchIterator.init(split.train_indices.items, 4);
    const b1 = iter.next() orelse unreachable;
    const b2 = iter.next() orelse unreachable;
    const b3 = iter.next();

    try std.testing.expectEqual(@as(usize, 4), b1.len);
    try std.testing.expectEqual(@as(usize, 2), b2.len);
    try std.testing.expect(b3 == null);
}

test "data loader supports stratified batches and caching/prefetch" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const csv_content =
        \\f1,f2,label
        \\1,10,0
        \\2,20,0
        \\3,30,1
        \\4,40,1
        \\5,50,0
        \\6,60,1
    ;
    try tmp.dir.writeFile(.{ .sub_path = "loader.csv", .data = csv_content });

    const cwd = std.fs.cwd();
    var prev_dir = try cwd.openDir(".", .{});
    defer prev_dir.close();
    try tmp.dir.setAsCwd();
    defer prev_dir.setAsCwd() catch {};

    var ds = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, "loader.csv", true);
    defer ds.deinit();

    var split = try lib.data_pipeline.splitIndices(allocator, ds.row_count, 1.0, 0.0, 1);
    defer split.deinit();

    var loader = try lib.data_pipeline.DataLoader.init(
        allocator,
        ds.asDataset(),
        split.train_indices.items,
        .{
            .batch_size = 2,
            .strategy = .stratified,
            .seed = 1,
            .prefetch = true,
            .use_cache = true,
        },
    );
    defer loader.deinit();

    const b1 = (try loader.next()) orelse unreachable;
    try std.testing.expectEqual(@as(usize, 2), b1.batch_len);
    try std.testing.expectEqual(@as(usize, 2), b1.feature_count);
    try std.testing.expectEqual(@as(usize, 2), b1.labels.len);
}

test "scalers fit and transform features" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const csv_content =
        \\f1,f2,label
        \\1,2,0
        \\3,4,1
        \\5,6,0
    ;
    try tmp.dir.writeFile(.{ .sub_path = "scale.csv", .data = csv_content });

    const cwd = std.fs.cwd();
    var prev_dir = try cwd.openDir(".", .{});
    defer prev_dir.close();
    try tmp.dir.setAsCwd();
    defer prev_dir.setAsCwd() catch {};

    var ds = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, "scale.csv", true);
    defer ds.deinit();

    const idx = [_]usize{ 0, 1, 2 };
    var std_scaler = try lib.data_pipeline.StandardScaler.fit(allocator, ds.asDataset(), &idx);
    defer std_scaler.deinit();
    var mm_scaler = try lib.data_pipeline.MinMaxScaler.fit(allocator, ds.asDataset(), &idx);
    defer mm_scaler.deinit();

    var row = [_]f32{ 3, 4 };
    std_scaler.transform(&row);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row[1], 1e-5);

    var row2 = [_]f32{ 3, 4 };
    mm_scaler.transform(&row2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), row2[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), row2[1], 1e-5);
}
