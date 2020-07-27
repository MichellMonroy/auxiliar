#include <math.h>
#include "element.h"
#include "population.h"
#include "recipes.h"

int integrate(Population population, Element **element, double begin, double end, double dt){
  int response = 0;
  int i,j,index;
  double a_x,a_y,a_z;
  double r,r3,r_x,r_y,r_z;
  double v_x,v_y,v_z;
  double dx,dy,dz;
  double G = 6.6743e-11; //m3 Kg-1 s-2
  index=0;
  for(i=begin;i<end; i++){
    //printf("index %i mass %lf x %lf\n",i,population.element[i].mass,population.element[i].position[0]);

    a_x=0.0;
    a_y=0.0;
    a_z=0.0;

    for(j=0;j < population.n_elements; j++){
      if (i!=j){
	r = compute_distance(population.element[i],population.element[j]);
	r3= pow(r,3.0);
	//printf("r3=%lf\n",r3);
 	r_x=population.element[j].position[0]-population.element[i].position[0]; 	
	r_y=population.element[j].position[1]-population.element[i].position[1]; 
	r_z=population.element[j].position[2]-population.element[i].position[2]; 
	a_x += G*population.element[j].mass*r_x/r3;
	a_y += G*population.element[j].mass*r_y/r3;
	a_z += G*population.element[j].mass*r_z/r3;
	//printf("G=%lf\tm=%lf\tr_x=%lf\tr3=%lf\ta_x=%le\n",G,population.element[j].mass,r_x,r3,a_x);
      }
    }
    v_x = population.element[i].velocity[0]+a_x*dt;
    v_y = population.element[i].velocity[1]+a_y*dt;
    v_z = population.element[i].velocity[2]+a_z*dt;
    
    



    dx = v_x*dt;
    dy = v_y*dt;
    dz = v_z*dt;

    //printf("mass %lf position %lf\n",population.element[i].mass,dx);
    (*element)[index].mass = population.element[i].mass;
    /*
    (*element)[index].position[0] += dx;
    (*element)[index].position[1] += dy;
    (*element)[index].position[2] += dz;

    (*element)[index].velocity[0] += v_x;
    (*element)[index].velocity[1] += v_y;
    (*element)[index].velocity[2] += v_z;
    */

    (*element)[index].position[0] = population.element[i].position[0]+ dx;
    (*element)[index].position[1] = population.element[i].position[1]+ dy;
    (*element)[index].position[2] = population.element[i].position[2]+ dz;

    (*element)[index].velocity[0] = v_x;
    (*element)[index].velocity[1] = v_y;
    (*element)[index].velocity[2] = v_z;

    //print_Element((*element)[index]);
    //printf("(%le,%le%le)\n",population.element[i].position[0]+dx,population.element[i].position[1]+dy,population.element[i].position[2]+dz);
    index++;
  }
  
  return response;
}
