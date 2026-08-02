#define _GNU_SOURCE
#include "vector_runtime.h"

#include <dirent.h>
#include <errno.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

int runtime_current_cpu(void) {
#if defined(__linux__)
    int cpu = sched_getcpu();
    return cpu >= 0 ? cpu : -1;
#else
    return -1;
#endif
}

int runtime_numa_node_of_cpu(int cpu) {
#if defined(__linux__)
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
    DIR *dir = opendir(path);
    if (dir == NULL) {
        return -1;
    }
    struct dirent *entry;
    int node = -1;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "node", 4) == 0) {
            int parsed = -1;
            if (sscanf(entry->d_name + 4, "%d", &parsed) == 1) {
                node = parsed;
                break;
            }
        }
    }
    closedir(dir);
    return node;
#else
    return -1;
#endif
}

int runtime_cpu_for_worker(int worker_index) {
    if (worker_index < 0) {
        return -1;
    }
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        return -1;
    }
    int seen = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &cpuset)) {
            if (seen == worker_index) {
                return cpu;
            }
            seen++;
        }
    }
    if (seen == 0) {
        return -1;
    }
    int target = worker_index % seen;
    seen = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &cpuset)) {
            if (seen == target) {
                return cpu;
            }
            seen++;
        }
    }
    return -1;
#elif defined(__APPLE__)
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count <= 0) {
        return worker_index;
    }
    return worker_index % (int) count;
#else
    return -1;
#endif
}

int runtime_memory_numa_node(const void *address, int64_t byte_size) {
#if defined(__linux__)
    if (address == NULL || byte_size <= 0) {
        return -1;
    }
#if defined(SYS_move_pages)
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return -1;
    }
    uintptr_t raw = (uintptr_t) address;
    void *page = (void *) (raw & ~((uintptr_t) page_size - 1));
    void *pages[1] = { page };
    int status[1] = { -1 };
    long rc = syscall(SYS_move_pages, 0, 1, pages, NULL, status, 0);
    if (rc == 0 && status[0] >= 0) {
        return status[0];
    }
#endif
    uintptr_t target = (uintptr_t) address;
    FILE *file = fopen("/proc/self/numa_maps", "r");
    if (file == NULL) {
        return -1;
    }
    char line[4096];
    char candidate[4096] = {0};
    uintptr_t candidate_start = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        char *end = NULL;
        uintptr_t start = (uintptr_t) strtoull(line, &end, 16);
        if (end == line) {
            continue;
        }
        if (start > target) {
            break;
        }
        candidate_start = start;
        strncpy(candidate, line, sizeof(candidate) - 1);
        candidate[sizeof(candidate) - 1] = '\0';
    }
    fclose(file);
    if (candidate_start == 0 || candidate[0] == '\0') {
        return -1;
    }
    int best_node = -1;
    long best_pages = -1;
    char *p = candidate;
    while ((p = strchr(p, 'N')) != NULL) {
        p++;
        if (!isdigit((unsigned char) *p)) {
            continue;
        }
        char *node_end = NULL;
        long node = strtol(p, &node_end, 10);
        if (node_end == NULL || *node_end != '=') {
            p = node_end == NULL ? p : node_end;
            continue;
        }
        char *pages_end = NULL;
        long pages = strtol(node_end + 1, &pages_end, 10);
        if (pages > best_pages) {
            best_pages = pages;
            best_node = (int) node;
        }
        p = pages_end == NULL ? node_end + 1 : pages_end;
    }
    return best_node;
#else
    return -1;
#endif
}

int runtime_pin_current_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0 ? 1 : 0;
#elif defined(__APPLE__)
    int qos_ok = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) == 0;
    thread_affinity_policy_data_t policy = { cpu + 1 };
    kern_return_t affinity_rc = thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY,
        (thread_policy_t) &policy, THREAD_AFFINITY_POLICY_COUNT);
    return (qos_ok || affinity_rc == KERN_SUCCESS) ? 1 : 0;
#else
    return 0;
#endif
}
