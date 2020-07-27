#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdio.h>
#include "element.h"
#include "population.h"

  int integrate(Population population,Element **element, double begin, double end, double dt);
  
#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* INTEGRATOR_H */

