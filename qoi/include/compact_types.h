#ifndef COMPACT_TYPES
#define COMPACT_TYPES

typedef struct instance_t instance_t;
typedef struct platform_t platform_t;

const char *get_error_msg(int error);
void get_platforms(instance_t *ins, int n_entries);

#endif