// Start of backends/webgpu.h.

// Synchronous wrapper around asynchronous WebGPU APIs, based on looping with
// emscripten_sleep until the respective callback gets called.

struct wgpu_wait_info {
  bool released;
  void *result;
};

void wgpu_map_sync_callback(WGPUBufferMapAsyncStatus status, void *info_v) {
  wgpu_wait_info *info = (wgpu_wait_info *)info_v;
  *info->result = status;
  info->released = true;
}

WGPUBufferMapAsyncStatus wgpu_map_buffer_sync(WGPUBuffer buffer,
                                              WGPUMapModeFlags mode,
                                              size_t offset, size_t size) {
  WGPUBufferMapAsyncStatus status;
  wgpu_wait_info info = {
    .released = false,
    .result = (void *)&status,
  };

  wgpuBufferMapAsync(buffer, mode, offset, size,
                     wgpu_map_sync_callback, (void *) &info);

  // TODO: Should this do some kind of volatile load?
  // (Same for other _sync wrappers below.)
  while (!info.released) {
    emscripten_sleep(0);
  }

  return status;
}

struct wgpu_request_adapter_result {
  WGPURequestAdapterStatus status;
  WGPUAdapter adapter;
  char *message;
};

void wgpu_request_adapter_callback(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter,
                                   const char *message, void *userdata) {
  wgpu_wait_info *info = (wgpu_wait_info *)info_v;
  wgpu_request_adapter_result *result
    = (wgpu_request_adapter_result *)info->result;
  result->status = status;
  result->adapter = adapter;
  result->message = message;
  info->released = true;
}

wgpu_request_adapter_result wgpu_request_adapter_sync(
    WGPUInstance instance, WGPURequestAdapterOptions const * options) {
  wgpu_request_adapter_result result = {};
  wgpu_wait_info info = {
    .released = false,
    .result = (void *)&result,
  };

  wgpuInstanceRequestAdapter(instance, options, wgpu_request_adapter_callback,
                             (void *)&info;

  while (!info.released) {
    emscripten_sleep(0);
  }

  return result;
}

struct wgpu_request_device_result {
  WGPURequestDeviceStatus status;
  WGPUDevice device;
  char *message;
};

void wgpu_request_device_callback(WGPURequestDeviceStatus status,
                                   WGPUDevice device,
                                   const char *message, void *userdata) {
  wgpu_wait_info *info = (wgpu_wait_info *)info_v;
  wgpu_request_device_result *result
    = (wgpu_request_device_result *)info->result;
  result->status = status;
  result->device = device;
  result->message = message;
  info->released = true;
}

wgpu_request_device_result wgpu_request_device_sync(
    WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor) {
  wgpu_request_device_result result = {};
  wgpu_wait_info info = {
    .released = false,
    .result = (void *)&result,
  };

  wgpuAdapterRequestDevice(adapter, descriptor, wgpu_request_device_callback,
                           (void *)&info;

  while (!info.released) {
    emscripten_sleep(0);
  }

  return result;
}

struct futhark_context_config {
  int in_use;
  int debugging;
  int profiling;
  int logging;
  char *cache_fname;
  int num_tuning_params;
  int64_t *tuning_params;
  const char** tuning_param_names;
  const char** tuning_param_vars;
  const char** tuning_param_classes;
  // Uniform fields above.
  
  char *program;
};

static void backend_context_config_setup(struct futhark_context_config *cfg) {
  cfg->program = strconcat(gpu_program);
}

static void backend_context_config_teardown(struct futhark_context_config *cfg) {
  free(cfg->program);
}

const char* futhark_context_config_get_program(struct futhark_context_config *cfg) {
  return cfg->program;
}

void futhark_context_config_set_program(struct futhark_context_config *cfg, const char *s) {
  free(cfg->program);
  cfg->program = strdup(s);
}

struct futhark_context {
  struct futhark_context_config* cfg;
  int detail_memory;
  int debugging;
  int profiling;
  int profiling_paused;
  int logging;
  lock_t lock;
  char *error;
  lock_t error_lock;
  FILE *log;
  struct constants *constants;
  struct free_list free_list;
  struct event_list event_list;
  int64_t peak_mem_usage_default;
  int64_t cur_mem_usage_default;
  struct program* program;
  bool program_initialised;
  // Uniform fields above.
 
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  // One module contains all the kernels as separate entry points.
  WGPUShaderModule module;

  struct free_list gpu_free_list;

  size_t lockstep_width;

  struct builtin_kernels* kernels;
};

int backend_context_setup(struct futhark_context *ctx) {
  ctx->kernels = NULL;

  ctx->instance = wgpuCreateInstance(NULL);

  wgpu_request_adapter_result adapter_result
    = wgpu_request_adapter_sync(ctx->instance, NULL);
  if (adapter_result.status != WGPURequestAdapterStatus_Success) {
    if (adapter_result.message != NULL) {
      futhark_panic(-1, "Could not get WebGPU adapter, status: %d\nMessage: %s\n",
                    adapter_result.status, adapter_result.message);
    } else {
      futhark_panic(-1, "Could not get WebGPU adapter, status: %d\n",
                    adapter_result.status);
    }
  }
  ctx->adapter = adapter_result.adapter;

  wgpu_request_device_result device_result
    = wgpu_request_device_sync(ctx->adapter, NULL);
  if (device_result.status != WGPURequestDeviceStatus_Success) {
    if (device_result.message != NULL) {
      futhark_panic(-1, "Could not get WebGPU device, status: %d\nMessage: %s\n",
                    device_result.status, device_result.message);
    } else {
      futhark_panic(-1, "Could not get WebGPU device, status: %d\n",
                    device_result.status);
    }
  }
  ctx->device = device_result.device;

  ctx->queue = wgpuDeviceGetQueue(ctx->device);
}

// GPU ABSTRACTION LAYER

// Types.
struct wgpu_kernel {
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
};
typedef struct wgpu_kernel* gpu_kernel;
typedef WGPUBuffer gpu_mem;

static void gpu_create_kernel(struct futhark_context *ctx,
                              gpu_kernel *kernel,
                              const char *name) {
  if (ctx->debugging) {
    fprintf(ctx->log, "Creating kernel %s.\n", name);
  }

  // TODO: Need layout information here to create pipeline etc.
}

static void gpu_free_kernel(struct futhark_context *ctx,
                            gpu_kernel kernel) {
  (void)ctx;
  (void)kernel;
}

static int gpu_scalar_to_device(struct futhark_context *ctx,
                                gpu_mem dst, size_t offset, size_t size,
                                void *src) {
  wgpuQueueWriteBuffer(ctx->queue, dst, offset, src, size);
}

static int gpu_scalar_from_device(struct futhark_context *ctx,
                                  void *dst,
                                  gpu_mem src, size_t offset, size_t size) {
  // TODO: It would probably be nice to re-use a buffer here instead.
  // Is there a guarantee for a maximum size when the _scalar_ functions are
  // used?
  WGPUBufferDescriptor desc = {
    .label = "tmp_readback",
    .size = size,
    .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
  };
  WGPUBuffer readback = wgpuDeviceCreateBuffer(ctx->device, &desc);
  
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(encoder,
    src, offset,
    readback, 0,
    size);

  WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &commandBuffer);

  WGPUBufferMapAsyncStatus status = 
    wgpu_map_buffer_sync(readback, WGPUMapMode_Read, 0, size);
  if (status != WGPUBufferMapAsyncStatus_Success) {
    futhark_panic(-1, "Failed to read scalar from device memory with error %d\n",
                  status);
  }

  const void *mapped = wgpuBufferGetConstMappedRange(readback, 0, size);
  memcpy(dst, mapped, size);

  wgpuBufferUnmap(readback);
  wgpuBufferDestroy(readback);
  return FUTHARK_SUCCESS;
}

static int gpu_memcpy(struct futhark_context *ctx,
                      gpu_mem dst, int64_t dst_offset,
                      gpu_mem src, int64_t src_offset,
                      int64_t nbytes) {
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(encoder,
      src, src_offset, dst, dst_offset, nbytes);
  WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &commandBuffer);
  return FUTHARK_SUCCESS:
}

static int memcpy_host2gpu(struct futhark_context *ctx, bool sync,
                           gpu_mem dst, int64_t dst_offset,
                           const unsigned char *src, int64_t src_offset,
                           int64_t nbytes) {
  if (nbytes <= 0) { return FUTHARK_SUCCESS; }

  // There is no async copy to device memory at the moment (the spec for
  // `writeBuffer` specifies that a copy of the data is always made and there is
  // no other good option to use here), so we ignore the sync parameter.
  (void)sync;

  wgpuQueueWriteBuffer(ctx->queue, dst, dst_offset, src, src_offset, nbytes);
}

static int memcpy_gpu2host(struct futhark_context *ctx, bool sync,
                           unsigned char *dst, int64_t dst_offset,
                           gpu_mem src, int64_t src_offset,
                           int64_t nbytes) {
  if (nbytes <= 0) { return FUTHARK_SUCCESS; }

  WGPUBufferDescriptor desc = {
    .label = "tmp_readback",
    .size = nbytes,
    .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
  };
  WGPUBuffer readback = wgpuDeviceCreateBuffer(ctx->device, &desc);
  
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(encoder,
    src, offset,
    readback, 0,
    nbytes);

  WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &commandBuffer);

  // TODO: Could we do an actual async mapping here if `sync` is false?
  WGPUBufferMapAsyncStatus status = 
    wgpu_map_buffer_sync(readback, WGPUMapMode_Read, 0, nbytes);
  if (status != WGPUBufferMapAsyncStatus_Success) {
    futhark_panic(-1, "Failed to copy from device memory with error %d\n",
                  status);
  }

  const void *mapped = wgpuBufferGetConstMappedRange(readback, 0, nbytes);
  memcpy(dst, mapped, nbytes);

  wgpuBufferUnmap(readback);
  wgpuBufferDestroy(readback);
  return FUTHARK_SUCCESS;
}

static int gpu_launch_kernel(struct futhark_context* ctx,
                             gpu_kernel kernel, const char *name,
                             const int32_t grid[3],
                             const int32_t block[3],
                             unsigned int shared_mem_bytes,
                             int num_args,
                             void* args[num_args],
                             size_t args_sizes[num_args]) {
  // This is going to be a bit interesting.
  // With current WGSL generation, all scalar arguments are passed as one
  // uniform buffer, while all memory arguments are bound as part of a
  // WGPUBindGroup. That... doesn't seem to fit this interface super well.
  // I think this ties into `gpu_create_kernel` a lot. Again the two basic
  // options are:
  // 1. Generate some data structure that allows us to look up which arguments
  //    (by index) go to which offset in the scalars struct / which bind group
  //    index.
  // 2. Generate e.g. a function per kernel that does this mapping more
  //    directly. I guess then we can still implement this function if generate
  //    a table of kernel (name) -> that function.
}

static int gpu_alloc_actual(struct futhark_context *ctx,
                            size_t size, gpu_mem *mem_out) {
  WGPUBufferDescriptor desc = {
    .size = size,
    .usage = WGPUBufferUsage_CopySrc
           | WGPUBufferUsage_CopyDst
           | WGPUBufferUsage_Storage,
  };
  *mem_out = wgpuDeviceCreateBuffer(ctx->device, &desc);
  return FUTHARK_SUCCESS;
}

static int gpu_free_actual(struct futhark_context *ctx, gpu_mem mem) {
  (void)ctx;
  wgpuBufferDestroy(mem);
  return FUTHARK_SUCCESS;
}

// End of backends/webgpu.h.
