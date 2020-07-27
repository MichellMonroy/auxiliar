#include <stdio.h>
#include <stdlib.h>
#include "element.h"
#include "population.h"

int total_sum(int n){
  return n*(n-1)/2;
}

Population new_Population(char *name, int size){
  Population population;
  sprintf(population.name, "%s" , name);
  population.n_elements=0;
  population.size = size;
  population.element = malloc(size * sizeof(Element));
  //printf("malloc %i\n",total_sum(size));
  population.n_distance = total_sum(size);
  population.distance = malloc(population.n_distance*sizeof(double));

  return population;

}

int add_element_to_population(Population *population, Element element){
  population->element[population->n_elements] = element;
  population->n_elements++;
  return population->n_elements;
}

int print_Population(Population population){
  int i;
  int n;
  //printf("%i\n",population.n_elements);
  for(i=0;i<population.n_elements;i++){
    print_Element(population.element[i]);
  }
  //n = sizeof(population.distance)/sizeof(float);  
  /*
  printf("List of distances %i\n",population.n_distance);
  for(i=0;i< population.n_distance ;i++){
    printf("%lf\t",population.distance[i]);
  }
  printf("\n");
  */
  
  return 1;
}

int print_distances(Population population, int begin, int end){
  int i;
  int n;
  
  //n = sizeof(population.distance)/sizeof(float);  
  
  //printf("List of distances %i\n",population.n_distance);
  for(i=begin;i < end ;i++){
    printf("%lf\t",population.distance[i]);
  }
  printf("\n"); 
  return 1;
}

int write_distances(char *path,Population population, int begin, int end, int miproc){
  int i;
  int n;
  FILE *f;
  char filename[256];
  //n = sizeof(population.distance)/sizeof(float);  
  //printf("List of distances %i\n",population.n_distance);
  sprintf(filename,"%s/distances-%i.dat",path,miproc);
  f = fopen(filename,"w");
  for(i=begin;i < end ;i++){
    fprintf(f,"%i\t%lf\n",i,population.distance[i]);
  }
  fflush(f);
  fclose(f);
  //printf("\n"); 
  return 1;
}


