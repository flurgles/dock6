#ifndef __APPLE__

#include <stdio.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <unistd.h>

// MEMORY FUNCTIONS
int parseMemLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++; 
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

// this is VIRT RAM MEMORY
int getVirtValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseMemLine(line);
            break;
        }
    }    
    fclose(file);
    return result;
}

// this is PHYSICAL RAM memory
int getPhysValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseMemLine(line);
            break;
        }
    }    
    fclose(file);
    return result;
}

static long readMemInfoKeyKB(const char *key)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    size_t klen = strlen(key);
    long out = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, klen) == 0) {
            char *p = line + klen;
            while (*p && (*p < '0' || *p > '9')) p++;
            if (*p) out = atol(p);
            break;
        }
    }
    fclose(f);
    return out;
}

long getMemAvailableKB(void)
{
    long v = readMemInfoKeyKB("MemAvailable:");
    if (v < 0) v = readMemInfoKeyKB("MemFree:");
    return v;
}

long getMemFreeKB(void)
{
    return readMemInfoKeyKB("MemFree:");
}

long getSwapFreeKB(void)
{
    return readMemInfoKeyKB("SwapFree:");
}

int hostMemoryLow(int threshold_mb)
{
    long avail = getMemAvailableKB();
    if (avail < 0) return -1;
    return (avail / 1024) < threshold_mb ? 1 : 0;
}
#endif
