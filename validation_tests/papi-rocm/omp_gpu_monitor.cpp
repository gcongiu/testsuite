#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include "matmul.h"

int main(int argc, char *argv[])
{
    int papi_errno;
    hipError_t hip_errno;

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        exit(EXIT_FAILURE);
    }

    papi_errno = PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num);
    if (papi_errno != PAPI_OK) {
        exit(EXIT_FAILURE);
    }

#define NUM_EVENTS 2
    const char *events[NUM_EVENTS] = {
        "rocm:::SQ_WAVES",
        "rocm:::SQ_WAVES_RESTORED",
    };

#pragma omp parallel
    {
        int eventset = PAPI_NULL;
        int thread_num = omp_get_thread_num();

        papi_errno = PAPI_create_eventset(&eventset);
        if (papi_errno != PAPI_OK) {
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < NUM_EVENTS; ++j) {
            char named_event[PAPI_MAX_STR_LEN];
            sprintf(named_event, "%s:device=%d", events[j], thread_num);
            papi_errno = PAPI_add_named_event(eventset, (const char*) named_event);
            if (papi_errno != PAPI_OK && papi_errno != PAPI_ENOEVNT) {
                exit(EXIT_FAILURE);
            }
        }

        papi_errno = PAPI_start(eventset);
        if (papi_errno != PAPI_OK) {
            exit(EXIT_FAILURE);
        }

        hip_errno = hipSetDevice(thread_num);
        if (hip_errno != hipSuccess) {
            exit(EXIT_FAILURE);
        }

        hipStream_t stream;
        hip_errno = hipStreamCreate(&stream);
        if (hip_errno != hipSuccess) {
            exit(EXIT_FAILURE);
        }

        void *handle;
        int matmul_errno;
        matmul_errno = matmul_init(&handle);
        if (matmul_errno != MATMUL_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        matmul_errno = matmul_run(handle, stream);
        if (matmul_errno != MATMUL_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        hip_errno = hipStreamSynchronize(stream);
        if (hip_errno != hipSuccess) {
            exit(EXIT_FAILURE);
        }

        hip_errno = hipStreamDestroy(stream);
        if (hip_errno != hipSuccess) {
            exit(EXIT_FAILURE);
        }

        matmul_errno = matmul_finalize(&handle);
        if (matmul_errno != MATMUL_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        long long counters[NUM_EVENTS] = { 0 };
        papi_errno = PAPI_stop(eventset, counters);
        if (papi_errno != PAPI_OK) {
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < NUM_EVENTS; ++i) {
            fprintf(stdout, "[tid:%d] %s:device=%d : %lld\n",
                    omp_get_thread_num(), events[i], thread_num,
                    counters[i]);
        }

        papi_errno = PAPI_cleanup_eventset(eventset);
        if (papi_errno != PAPI_OK) {
            exit(EXIT_FAILURE);
        }

        papi_errno = PAPI_destroy_eventset(&eventset);
        if (papi_errno != PAPI_OK) {
            exit(EXIT_FAILURE);
        }
    }

    PAPI_shutdown();

    return EXIT_SUCCESS;
}
