/*******************************************************************************************************************
* FILE NAME   :    tools_metric.h
*
* PROJECTION  :    general purpose
*
* DESCRIPTION :    various metric tools
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2022 APR 05      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once

#include "all_common.h"
#include <omp.h>
#include <thread>
#include <chrono>
#include <Windows.h>

namespace metric
{
    using pt = std::chrono::steady_clock::time_point;  // time point
    using rt = std::chrono::steady_clock;              // time root
    using ms = std::chrono::milliseconds;              // 
    using us = std::chrono::microseconds;              //
    using ns = std::chrono::nanoseconds;               //
    // metric::pt begin = metric::rt::now();
    // metric::pt   end = metric::rt::now();
    // std::chrono::duration_cast<metric::us>(end - begin).count();

    //================================
    inline void max_process(void)
    {
        printf("maximum processes in openmp: %d\n", omp_get_num_procs());
        printf("maximum threads  in  openmp: %d\n", omp_get_max_threads());
        printf("maximum threads in hardware: %d\n", std::thread::hardware_concurrency());
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        printf("maximum processors in win32: %d\n", (int)sysinfo.dwNumberOfProcessors);
    }
}