#ifndef VECTOR_RUNTIME_H
#define VECTOR_RUNTIME_H

#include <stdint.h>

int runtime_current_cpu(void);
int runtime_numa_node_of_cpu(int cpu);
int runtime_cpu_for_worker(int worker_index);
int runtime_memory_numa_node(const void *address, int64_t byte_size);
int runtime_pin_current_thread(int cpu);

#endif
