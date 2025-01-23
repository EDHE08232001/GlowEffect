/*******************************************************************************************************************
 * FILE NAME   :    tools_cuda_dinfo.h
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    check gpu device
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 MAR 05      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

inline void cuda_device_info(const int n_print = 1)
{
	int n_device;
	cudaGetDeviceCount(&n_device);

	for (int k = 0; k < n_device; k++)
	{
		cudaSetDevice(k);
		int device = -1;
		cudaGetDevice(&device);
		if (device < 0)
			exit(-1);

		cudaDeviceProp d_props;
		cudaGetDeviceProperties(&d_props, device);

		printf("\ndevice name: %s\n", d_props.name);

		if (!n_print)
			continue;
		printf("global mem: %lld\n", d_props.totalGlobalMem);
		printf("shared mem per block: %lld\n", d_props.sharedMemPerBlock);
		printf("regs per block: %d\n", d_props.regsPerBlock);
		printf("warp size: %d\n", d_props.warpSize);
		printf("mem pitch: %lld\n", d_props.memPitch);
		printf("maxThreadsPerBlock: %d\n", d_props.maxThreadsPerBlock);
		printf("maxThreadsDim: %d %d %d\n", d_props.maxThreadsDim[0], d_props.maxThreadsDim[1], d_props.maxThreadsDim[2]);
		printf("maxGridSize: %d %d %d\n", d_props.maxGridSize[0], d_props.maxGridSize[1], d_props.maxGridSize[2]);
		printf("clockRate: %d\n", d_props.clockRate);
		printf("totalConstMem: %lld\n", d_props.totalConstMem);
		printf("compute capability %d%d\n", d_props.major, d_props.minor);
		printf("textureAlignment: %lld\n", d_props.textureAlignment);
		printf("texturePitchAlignment: %lld\n", d_props.texturePitchAlignment);
		printf("deviceOverlap: %d\n", d_props.deviceOverlap);
		printf("multiProcessorCount: %d\n", d_props.multiProcessorCount);
		printf("kernelExecTimeoutEnabled: %d\n", d_props.kernelExecTimeoutEnabled);
		printf("integrated: %d\n", d_props.integrated);
		printf("canMapHostMemory: %d\n", d_props.canMapHostMemory);
		printf("computeMode: %d\n", d_props.computeMode);
		printf("maxTexture1D: %d\n", d_props.maxTexture1D);
		printf("maxTexture1DMipmap: %d\n", d_props.maxTexture1DMipmap);
		printf("maxTexture1DLinear: %d\n", d_props.maxTexture1DLinear);
		printf("maxTexture2D: %d %d\n", d_props.maxTexture2D[0], d_props.maxTexture2D[1]);
		printf("maxTexture2DMipmap: %d %d\n", d_props.maxTexture2DMipmap[0], d_props.maxTexture2DMipmap[1]);
		printf("maxTexture2DLinear: %d %d %d\n", d_props.maxTexture2DLinear[0], d_props.maxTexture2DLinear[1], d_props.maxTexture2DLinear[2]);
		printf("maxTexture2DGather: %d %d\n", d_props.maxTexture2DGather[0], d_props.maxTexture2DGather[1]);
		printf("maxTexture3D: %d %d %d\n", d_props.maxTexture3D[0], d_props.maxTexture3D[1], d_props.maxTexture3D[2]);
		printf("maxTexture3DAlt: %d %d %d\n", d_props.maxTexture3DAlt[0], d_props.maxTexture3DAlt[1], d_props.maxTexture3DAlt[2]);
		printf("maxTextureCubemap: %d\n", d_props.maxTextureCubemap);
		printf("maxTexture1DLayered: %d %d\n", d_props.maxTexture1DLayered[0], d_props.maxTexture1DLayered[1]);
		printf("maxTexture2DLayered: %d %d %d\n", d_props.maxTexture2DLayered[0], d_props.maxTexture2DLayered[1], d_props.maxTexture2DLayered[2]);
		printf("maxTextureCubemapLayered: %d %d\n", d_props.maxTextureCubemapLayered[0], d_props.maxTextureCubemapLayered[1]);
		printf("maxSurface1D: %d\n", d_props.maxSurface1D);
		printf("maxSurface2D: %d %d\n", d_props.maxSurface2D[0], d_props.maxSurface2D[1]);
		printf("maxSurface3D: %d %d %d\n", d_props.maxSurface3D[0], d_props.maxSurface3D[1], d_props.maxSurface3D[2]);
		printf("maxSurface1DLayered: %d %d\n", d_props.maxSurface1DLayered[0], d_props.maxSurface1DLayered[1]);
		printf("maxSurface2DLayered: %d %d %d\n", d_props.maxSurface2DLayered[0], d_props.maxSurface2DLayered[1], d_props.maxSurface2DLayered[2]);
		printf("maxSurfaceCubemap: %d\n", d_props.maxSurfaceCubemap);
		printf("maxSurfaceCubemapLayered: %d %d\n", d_props.maxSurfaceCubemapLayered[0], d_props.maxSurfaceCubemapLayered[1]);
		printf("surfaceAlignment: %lld\n", d_props.surfaceAlignment);
		printf("concurrentKernels: %d\n", d_props.concurrentKernels);
		printf("ECCEnabled: %d\n", d_props.ECCEnabled);
		printf("pciBusID: %d\n", d_props.pciBusID);
		printf("pciDeviceID: %d\n", d_props.pciDeviceID);
		printf("pciDomainID: %d\n", d_props.pciDomainID);
		printf("tccDriver: %d\n", d_props.tccDriver);
		printf("asyncEngineCount: %d\n", d_props.asyncEngineCount);
		printf("unifiedAddressing: %d\n", d_props.unifiedAddressing);
		printf("memoryClockRate: %d\n", d_props.memoryClockRate);
		printf("memoryBusWidth: %d\n", d_props.memoryBusWidth);
		printf("l2CacheSize: %d\n", d_props.l2CacheSize);
		printf("persistingL2CacheMaxSize: %d\n", d_props.persistingL2CacheMaxSize);
		printf("maxThreadsPerMultiProcessor: %d\n", d_props.maxThreadsPerMultiProcessor);
		printf("streamPrioritiesSupported: %d\n", d_props.streamPrioritiesSupported);
		printf("globalL1CacheSupported: %d\n", d_props.globalL1CacheSupported);
		printf("localL1CacheSupported: %d\n", d_props.localL1CacheSupported);
		printf("sharedMemPerMultiprocessor: %lld\n", d_props.sharedMemPerMultiprocessor);
		printf("regsPerMultiprocessor: %d\n", d_props.regsPerMultiprocessor);
		printf("managedMemory: %d\n", d_props.managedMemory);
		printf("isMultiGpuBoard: %d\n", d_props.isMultiGpuBoard);
		printf("multiGpuBoardGroupID: %d\n", d_props.multiGpuBoardGroupID);
		printf("hostNativeAtomicSupported: %d\n", d_props.hostNativeAtomicSupported);
		printf("singleToDoublePrecisionPerfRatio: %d\n", d_props.singleToDoublePrecisionPerfRatio);
		printf("pageableMemoryAccess: %d\n", d_props.pageableMemoryAccess);
		printf("concurrentManagedAccess: %d\n", d_props.concurrentManagedAccess);
		printf("computePreemptionSupported: %d\n", d_props.computePreemptionSupported);
		printf("canUseHostPointerForRegisteredMem: %d\n", d_props.canUseHostPointerForRegisteredMem);
		printf("cooperativeLaunch: %d\n", d_props.cooperativeLaunch);
		printf("cooperativeMultiDeviceLaunch: %d\n", d_props.cooperativeMultiDeviceLaunch);
		printf("sharedMemPerBlockOptin: %lld\n", d_props.sharedMemPerBlockOptin);
		printf("pageableMemoryAccessUsesHostPageTables: %d\n", d_props.pageableMemoryAccessUsesHostPageTables);
		printf("directManagedMemAccessFromHost: %d\n", d_props.directManagedMemAccessFromHost);
		printf("maxBlocksPerMultiProcessor: %d\n", d_props.maxBlocksPerMultiProcessor);
		printf("accessPolicyMaxWindowSize: %d\n", d_props.accessPolicyMaxWindowSize);
		printf("reservedSharedMemPerBlock: %lld\n", d_props.reservedSharedMemPerBlock);
	}
}


/*
* print out device limits
* 
* you can set these limits by cudaDeviceSetLimit(limit, value);
* 
*/
inline void cuda_device_limit()
{
	size_t d_value;
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitStackSize);
	printf("cudaLimitStackSize: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitPrintfFifoSize);
	printf("cudaLimitPrintfFifoSize: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitDevRuntimeSyncDepth);
	printf("cudaLimitDevRuntimeSyncDepth: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitDevRuntimePendingLaunchCount);
	printf("cudaLimitDevRuntimePendingLaunchCount: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitMaxL2FetchGranularity);
	printf("cudaLimitMaxL2FetchGranularity: %ld\n", (int)d_value);
	cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitPersistingL2CacheSize);
	printf("cudaLimitPersistingL2CacheSizes: %ld\n", (int)d_value);
}