// Start of builtin_kernels.wgsl

// Constants used for transpositions. In principle these should be configurable.
const TR_BLOCK_DIM:        i32 = 16;
const TR_TILE_DIM:         i32 = 32;
const TR_ELEMS_PER_THREAD: i32 = 8;

// Shared memory for transpositions.
// In shared memory, every i8 and i16 is represented by a full i32
var<workgroup> transpose_shared_memory_i8: array<i32, TR_TILE_DIM*(TR_TILE_DIM+1)>;
var<workgroup> transpose_shared_memory_i16: array<i32, TR_TILE_DIM*(TR_TILE_DIM+1)>;
var<workgroup> transpose_shared_memory_i32: array<i32, TR_TILE_DIM*(TR_TILE_DIM+1)>;
var<workgroup> transpose_shared_memory_i64: array<i64, TR_TILE_DIM*(TR_TILE_DIM+1)>;

override lmad_copy_block_size_x: i32 = 1;
override lmad_copy_block_size_y: i32 = 1;
override lmad_copy_block_size_z: i32 = 1;

override map_transpose_block_size_x: i32 = 1;
override map_transpose_block_size_y: i32 = 1;
override map_transpose_block_size_z: i32 = 1;

override map_transpose_low_height_block_size_x: i32 = 1;
override map_transpose_low_height_block_size_y: i32 = 1;
override map_transpose_low_height_block_size_z: i32 = 1;

override map_transpose_low_width_block_size_x: i32 = 1;
override map_transpose_low_width_block_size_y: i32 = 1;
override map_transpose_low_width_block_size_z: i32 = 1;

override map_transpose_small_block_size_x: i32 = 1;
override map_transpose_small_block_size_y: i32 = 1;
override map_transpose_small_block_size_z: i32 = 1;

override map_transpose_large_block_size_x: i32 = 1;
override map_transpose_large_block_size_y: i32 = 1;
override map_transpose_large_block_size_z: i32 = 1;

struct MapTransposeParameters {
    dst_offset: i64,    // 0
    src_offset: i64,    // 8
    num_arrays: i32,    // 16
    x_elems: i32,       // 20
    y_elems: i32,       // 24
    mulx: i32,          // 28
    muly: i32,          // 32
    repeat_1: i32,      // 36
    repeat_2: i32       // 40
}

struct MapTransposeParametersLarge {
    dst_offset: i64,    // 0
    src_offset: i64,    // 8
    num_arrays: i64,    // 16
    x_elems: i64,       // 24
    y_elems: i64,       // 32
    mulx: i64,          // 40
    muly: i64,          // 48
    repeat_1: i32,      // 56
    repeat_2: i32       // 60
}

struct CopyParameters {
    dst_offset: i64,
    src_offset: i64,
    n: i64,
    r: i32,
    shape0: i64, dst_stride0: i64, src_stride0: i64,
    shape1: i64, dst_stride1: i64, src_stride1: i64,
    shape2: i64, dst_stride2: i64, src_stride2: i64,
    shape3: i64, dst_stride3: i64, src_stride3: i64,
    shape4: i64, dst_stride4: i64, src_stride4: i64,
    shape5: i64, dst_stride5: i64, src_stride5: i64,
    shape6: i64, dst_stride6: i64, src_stride6: i64,
    shape7: i64, dst_stride7: i64, src_stride7: i64
}

// Begin of builtin kernel group: NAME, ELEM_TYPE

@group(0) @binding(0) var<uniform> copy_args_NAME: CopyParameters;
@group(0) @binding(1) var<storage, read_write> copy_dst_NAME_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> copy_src_NAME_mem: array<ELEM_TYPE>;
@compute @workgroup_size(lmad_copy_block_size_x, lmad_copy_block_size_y, lmad_copy_block_size_z)
fn lmad_copy_NAME(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var remainder: i32 = i32(global_id.x);
    var dst_offset: i32 = copy_args_NAME.dst_offset.x;
    var src_offset: i32 = copy_args_NAME.src_offset.x;

    if (i32(global_id.x) >= copy_args_NAME.n.x) {
      return;
    }

    if (copy_args_NAME.r > 0) {
      let i: i32 = remainder % copy_args_NAME.shape0.x;
      dst_offset += i * copy_args_NAME.dst_stride0.x;
      src_offset += i * copy_args_NAME.src_stride0.x;
      remainder  /= copy_args_NAME.shape0.x;
    }
    if (copy_args_NAME.r > 1) {
      let i: i32 = remainder % copy_args_NAME.shape1.x;
      dst_offset += i * copy_args_NAME.dst_stride1.x;
      src_offset += i * copy_args_NAME.src_stride1.x;
      remainder  /= copy_args_NAME.shape1.x;
    }
    if (copy_args_NAME.r > 2) {
      let i: i32 = remainder % copy_args_NAME.shape2.x;
      dst_offset += i * copy_args_NAME.dst_stride2.x;
      src_offset += i * copy_args_NAME.src_stride2.x;
      remainder  /= copy_args_NAME.shape2.x;
    }
    if (copy_args_NAME.r > 3) {
      let i: i32 = remainder % copy_args_NAME.shape3.x;
      dst_offset += i * copy_args_NAME.dst_stride3.x;
      src_offset += i * copy_args_NAME.src_stride3.x;
      remainder  /= copy_args_NAME.shape3.x;
    }
    if (copy_args_NAME.r > 4) {
      let i: i32 = remainder % copy_args_NAME.shape4.x;
      dst_offset += i * copy_args_NAME.dst_stride4.x;
      src_offset += i * copy_args_NAME.src_stride4.x;
      remainder  /= copy_args_NAME.shape4.x;
    }
    if (copy_args_NAME.r > 5) {
      let i: i32 = remainder % copy_args_NAME.shape5.x;
      dst_offset += i * copy_args_NAME.dst_stride5.x;
      src_offset += i * copy_args_NAME.src_stride5.x;
      remainder  /= copy_args_NAME.shape5.x;
    }
    if (copy_args_NAME.r > 6) {
      let i: i32 = remainder % copy_args_NAME.shape6.x;
      dst_offset += i * copy_args_NAME.dst_stride6.x;
      src_offset += i * copy_args_NAME.src_stride6.x;
      remainder  /= copy_args_NAME.shape6.x;
    }
    if (copy_args_NAME.r > 7) {
      let i: i32 = remainder % copy_args_NAME.shape7.x;
      dst_offset += i * copy_args_NAME.dst_stride7.x;
      src_offset += i * copy_args_NAME.src_stride7.x;
      remainder  /= copy_args_NAME.shape7.x;
    }

    write_ELEM_TYPE(&copy_dst_NAME_mem, dst_offset, read_ELEM_TYPE(&copy_src_NAME_mem, src_offset));
}

@group(0) @binding(0) var<uniform> mt_NAME_args: MapTransposeParameters;
@group(0) @binding(1) var<storage, read_write> mt_NAME_dst_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> mt_NAME_src_mem: array<ELEM_TYPE>;
@compute @workgroup_size(map_transpose_block_size_x, map_transpose_block_size_y, map_transpose_block_size_z)
fn map_transpose_NAME(
    @builtin(workgroup_id)         group_id: vec3<u32>,  // tblock_id -> unique id of a group  within a dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>, // global_id -> unique id of a thread within a dispatch
    @builtin(local_invocation_id)  local_id: vec3<u32>,  // local_id  -> unique id of a thread within a group
    @builtin(num_workgroups)       num_groups: vec3<u32>
) {
    let dst_offset = mt_NAME_args.dst_offset[0];
    let src_offset = mt_NAME_args.src_offset[0];

    let tblock_id_0 = i32(group_id[0]);
    let global_id_0 = i32(global_id[0]);
    var tblock_id_1 = i32(group_id[1]);
    var global_id_1 = i32(global_id[1]);

    for (var i1 = 0; i1 <= mt_NAME_args.repeat_1; i1++) {
        var tblock_id_2 = i32(group_id[2]);
        var global_id_2 = i32(global_id[2]);

        for (var i2 = 0; i2 < mt_NAME_args.repeat_2; i2++) {
            var our_array_offset = tblock_id_2 * mt_NAME_args.x_elems * mt_NAME_args.y_elems;
            var odata_offset = dst_offset + our_array_offset;
            var idata_offset = src_offset + our_array_offset;
            var x_index = i32(global_id_0);
            var y_index = tblock_id_1 * TR_TILE_DIM + i32(local_id[1]);

            if (x_index < mt_NAME_args.x_elems) {
                for (var j = 0; j < TR_ELEMS_PER_THREAD; j++) {
                    var index_i = (y_index + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD)) * mt_NAME_args.x_elems + x_index;
                    if (y_index + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD) < mt_NAME_args.y_elems) {
                        let shared_offset = (i32(local_id[1]) + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD)) * (TR_TILE_DIM+1) + i32(local_id[0]);
                        let src_val = read_ELEM_TYPE(&mt_NAME_src_mem, idata_offset + index_i);
                        transpose_shared_memory_ELEM_TYPE[shared_offset] = src_val;
                    }
                }
            }

            workgroupBarrier();

            x_index = tblock_id_1 * TR_TILE_DIM + i32(local_id[0]);
            y_index = tblock_id_0 * TR_TILE_DIM + i32(local_id[1]);

            if (x_index < mt_NAME_args.y_elems) {
                for (var j = 0; j < TR_ELEMS_PER_THREAD; j++) {
                    var index_out = (y_index + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD)) * mt_NAME_args.y_elems + x_index;
                    if (y_index + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD) < mt_NAME_args.x_elems) {
                        let shared_offset = i32(local_id[0]) * (TR_TILE_DIM+1) + i32(local_id[1]) + j * (TR_TILE_DIM/TR_ELEMS_PER_THREAD);
                        let src_val = ELEM_TYPE(transpose_shared_memory_ELEM_TYPE[shared_offset]);
                        write_ELEM_TYPE(&mt_NAME_dst_mem, odata_offset + index_out, src_val);
                    }
                }
            }

            tblock_id_2 += i32(num_groups[2]);
            global_id_2 += i32(num_groups[2] * num_groups[2]);
        }

        tblock_id_1 += i32(num_groups[1]);
        global_id_1 += i32(num_groups[1] * num_groups[1]);
    }
}

@group(0) @binding(0) var<uniform> mt_lh_NAME_args: MapTransposeParameters;
@group(0) @binding(1) var<storage, read_write> mt_lh_NAME_dst_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> mt_lh_NAME_src_mem: array<ELEM_TYPE>;
@compute @workgroup_size(map_transpose_low_height_block_size_x, map_transpose_low_height_block_size_y, map_transpose_low_height_block_size_z)
fn map_transpose_NAME_low_height(
    @builtin(workgroup_id)         group_id: vec3<u32>,  // tblock_id -> unique id of a group  within a dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>, // global_id -> unique id of a thread within a dispatch
    @builtin(local_invocation_id)  local_id: vec3<u32>,  // local_id  -> unique id of a thread within a group
    @builtin(num_workgroups)       num_groups: vec3<u32>
) {
    for (var y = 0; y < mt_lh_NAME_args.y_elems; y += 1) {
        for (var x = 0; x < mt_lh_NAME_args.x_elems; x += 1) {
            let dst_idx = mt_lh_NAME_args.dst_offset[0] + x * mt_lh_NAME_args.y_elems + y;
            let src_idx = mt_lh_NAME_args.src_offset[0] + y * mt_lh_NAME_args.y_elems + x;
            write_ELEM_TYPE(&mt_lh_NAME_dst_mem, dst_idx, read_ELEM_TYPE(&mt_lh_NAME_src_mem, src_idx));
        }
    }
}

@group(0) @binding(0) var<uniform> mt_lw_NAME_args: MapTransposeParameters;
@group(0) @binding(1) var<storage, read_write> mt_lw_NAME_dst_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> mt_lw_NAME_src_mem: array<ELEM_TYPE>;
@compute @workgroup_size(map_transpose_low_width_block_size_x, map_transpose_low_width_block_size_y, map_transpose_low_width_block_size_z)
fn map_transpose_NAME_low_width(
    @builtin(workgroup_id)         group_id: vec3<u32>,  // tblock_id -> unique id of a group  within a dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>, // global_id -> unique id of a thread within a dispatch
    @builtin(local_invocation_id)  local_id: vec3<u32>,  // local_id  -> unique id of a thread within a group
    @builtin(num_workgroups)       num_groups: vec3<u32>
) {
    for (var y = 0; y < mt_lw_NAME_args.y_elems; y += 1) {
        for (var x = 0; x < mt_lw_NAME_args.x_elems; x += 1) {
            let dst_idx = mt_lw_NAME_args.dst_offset[0] + x * mt_lw_NAME_args.y_elems + y;
            let src_idx = mt_lw_NAME_args.src_offset[0] + y * mt_lw_NAME_args.y_elems + x;
            write_ELEM_TYPE(&mt_lw_NAME_dst_mem, dst_idx, read_ELEM_TYPE(&mt_lw_NAME_src_mem, src_idx));
        }
    }
}

@group(0) @binding(0) var<uniform> mt_s_NAME_args: MapTransposeParameters;
@group(0) @binding(1) var<storage, read_write> mt_s_NAME_dst_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> mt_s_NAME_src_mem: array<ELEM_TYPE>;
@compute @workgroup_size(map_transpose_small_block_size_x, map_transpose_small_block_size_y, map_transpose_small_block_size_z)
fn map_transpose_NAME_small(
    @builtin(workgroup_id)         group_id: vec3<u32>,  // tblock_id -> unique id of a group  within a dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>, // global_id -> unique id of a thread within a dispatch
    @builtin(local_invocation_id)  local_id: vec3<u32>,  // local_id  -> unique id of a thread within a group
    @builtin(num_workgroups)       num_groups: vec3<u32>
) {
    for (var y = 0; y < mt_s_NAME_args.y_elems; y += 1) {
        for (var x = 0; x < mt_s_NAME_args.x_elems; x += 1) {
            let dst_idx = mt_s_NAME_args.dst_offset[0] + x * mt_s_NAME_args.y_elems + y;
            let src_idx = mt_s_NAME_args.src_offset[0] + y * mt_s_NAME_args.y_elems + x;
            write_ELEM_TYPE(&mt_s_NAME_dst_mem, dst_idx, read_ELEM_TYPE(&mt_s_NAME_src_mem, src_idx));
        }
    }
}

@group(0) @binding(0) var<uniform> mt_l_NAME_args: MapTransposeParametersLarge;
@group(0) @binding(1) var<storage, read_write> mt_l_NAME_dst_mem: array<ELEM_TYPE>;
@group(0) @binding(2) var<storage, read_write> mt_l_NAME_src_mem: array<ELEM_TYPE>;
@compute @workgroup_size(map_transpose_large_block_size_x, map_transpose_large_block_size_y, map_transpose_large_block_size_z)
fn map_transpose_NAME_large(
    @builtin(workgroup_id)         group_id: vec3<u32>,  // tblock_id -> unique id of a group  within a dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>, // global_id -> unique id of a thread within a dispatch
    @builtin(local_invocation_id)  local_id: vec3<u32>,  // local_id  -> unique id of a thread within a group
    @builtin(num_workgroups)       num_groups: vec3<u32>
) {
    for (var y = 0; y < mt_l_NAME_args.y_elems[0]; y += 1) {
        for (var x = 0; x < mt_l_NAME_args.x_elems[0]; x += 1) {
            let dst_idx = mt_l_NAME_args.dst_offset[0] + x * mt_l_NAME_args.y_elems[0] + y;
            let src_idx = mt_l_NAME_args.src_offset[0] + y * mt_l_NAME_args.y_elems[0] + x;
            write_ELEM_TYPE(&mt_l_NAME_dst_mem, dst_idx, read_ELEM_TYPE(&mt_l_NAME_src_mem, src_idx));
        }
    }
}

// End of builtin kernel group

// End of builtin_kernels.wgsl
