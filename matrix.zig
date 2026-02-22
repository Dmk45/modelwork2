const std = @import("std");
pub fn brew() ![5]i32 {
    var mut_array_of_int: [5]i32 = [5]i32{ 1, 2, 3, 4, 5 };
    mut_array_of_int[0] = 10;
    std.debug.print("mut_array_of_int: {any}\n", .{mut_array_of_int[0]});
    return mut_array_of_int;
}

pub fn main() !void {
    const x = try brew();
    std.debug.print("mut_array_of_int: {any}\n", .{x});
}
