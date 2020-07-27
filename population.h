#ifndef POPULATION_H
#define POPULATION_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "element.h"
  
  typedef struct{
    char name[64]; //name of the population
    int n_elements; //number of elements
    int size; //max number of elements
    Element *element; // List of elements
    double *distance; //List of distance between elements
    int n_distance; //number of computed distances
  }Population;

  Population new_Population(char *name, int size);
  int add_element_to_population(Population *population, Element element);
  int print_Population(Population population);
  int print_distances(Population population, int begin, int end);
  int write_distances(char *path,Population population, int begin, int end, int miproc);
  
#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* POPULATION_H */

