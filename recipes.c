#include <math.h>
#include <stdio.h>
#include <string.h>
#include "recipes.h"

double compute_distance(Element A, Element B){
  double x_a,x_b, y_a,y_b;
  x_a = A.position[0];
  y_a = A.position[1];
  x_b = B.position[0];
  y_b = B.position[1];
 return sqrt(pow(x_a-x_b,2) + pow(y_a-y_b,2)); 
}

unsigned long factorial(unsigned long f)
{
    if ( f == 0 ) 
        return 1;
    return(f * factorial(f - 1));
}


  
int write_block(FILE *in, FILE *out){
   char line[256];
   //printf("Starting write\n");
   while (fgets (line, 256, in) != EOF) {
     if (feof(in)) break;
     //if (strncmp(line,"#Time",5)==0){
     //  break;
     //}else{
       //printf("%s",line);
       if (strlen(line) >0){
	 if (line[0] != '#' ){
	   fprintf(out,"%s", line);
	 }
       }
       //printf("ok\n");
       //}
   }
   //printf("end\n");
   return 1;
}
