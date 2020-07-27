#include <stdio.h>
#include "element.h"

Element new_Element(double mass, double p0, double p1, double p2, double v0, double v1, double v2){
  Element element;
  element.mass = mass;
  element.position[0] = p0;
  element.position[1] = p1;
  element.position[2] = p2;
  
  element.velocity[0] = v0;
  element.velocity[1] = v1;
  element.velocity[2] = v2;
  return element;
}

int print_Element(Element element){
  printf("%le\t[%le,%le,%le]\t[%le,%le,%le]\n",element.mass, element.position[0],element.position[1],element.position[2],element.velocity[0],element.velocity[1],element.velocity[2]);
  return 1;
}

