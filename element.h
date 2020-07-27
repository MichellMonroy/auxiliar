#ifndef ELEMENT_H
#define ELEMENT_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  typedef struct{
    double mass; //kg
    double position[3]; //[x,y,z] m
    double velocity[3]; // [x,y,z] m/s
  }Element;

  Element new_Element(double mass, double p0, double p1, double p2, double v0, double v1, double v2);
  int print_Element(Element element);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* ELEMENT_H */

