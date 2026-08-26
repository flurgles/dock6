#ifndef _MEM_INFO_H_
#define _MEM_INFO_H_

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include "sys/types.h"


#ifndef __APPLE__
#include <sys/sysinfo.h>
#endif


int parseMemLine(char*);
int getVirtValue();
int getPhysValue();

/* Host memory availability helpers for VS window sizing and dynamic
   throttling.  All values in KB unless noted.  Returns -1 on error
   (e.g. /proc/meminfo unreadable or key missing). */
long getMemAvailableKB(void);
long getMemFreeKB(void);
long getSwapFreeKB(void);

/* 1 if available RAM is below threshold_mb (conservative pressure signal
   used to pause new VS dispatches before the kernel is forced to page).
   Returns 0 normally, -1 if availability could not be read. */
int hostMemoryLow(int threshold_mb);


#endif  
