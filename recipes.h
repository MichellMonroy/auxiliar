#ifndef RECIPES_H
#define RECIPES_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdio.h>
#include "element.h"

typedef int boolean;
#define true 1
#define false 0

  
  unsigned long factorial(unsigned long f);
  double compute_distance(Element A, Element B);
  int write_block(FILE *in, FILE *out);
  
#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* RECIPES_H */

