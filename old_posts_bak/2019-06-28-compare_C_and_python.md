---
title: "A Comparison of C and Python"
date: 2019-06-28
categories:
  - Data Science
tags: [Machine Learning Basics]
header:
  image: "/images/banners/banner_code.png"
excerpt: "Few examples in order to illustrate how to code in a Pythonic way"
mathjax: "true"
---


### Python module/package imports for this chapter


```python
import math
```

## Comparing C and Python: computing the digits of pi

Leibniz series: $\pi/4 = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots = \sum_{k=0} \frac{(-1)^k}{(2k+1)}$


```python
%%file pi.c

#include <math.h>
#include <stdio.h>

int main(int argc,char **argv) {
    int k;
    double acc = 0.0;
    
    for(k=0;k<10000;k++) {
        acc = acc + pow(-1,k)/(2*k+1);
    }
    
    acc = 4 * acc;
    
    printf("pi: %.15f\n",acc);
    
    return 0;
}
```

    Overwriting pi.c



```python
!gcc -o pi pi.c -lm
```

instead of gcc pi.c 
then ./a.out
The error you can see: error: ld returned 1 exit status is from the linker ld (part of gcc that combines the object files) because it is unable to find where the function pow is defined.
Including math.h brings in the declaration of the various functions and not their definition. The def is present in the math library libm.a. You need to link your program with this library so that the calls to functions like pow() are resolved.




```python
!./pi
```

    pi: 3.141492653590034


## In a python way


```python
acc = 0.0

for k in range(10000):
    acc = acc + pow(-1,k)/(2*k+1)

acc = 4 * acc

print("pi:",acc)
```

    pi: 3.1414926535900345


## In a more pythonic way


```python
4*sum(pow(-1,k)/(2*k+1) for k in range(10000))
```




    3.1414926535900345



#### With one line we got the same result: python is more synthetic and expressive


```python

```
