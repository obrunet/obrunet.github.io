---
title: "An exhaustive Python cheat sheet 1/2"
date: 2020-11-15
categories:
  - Pythonic ideas
tags: [Coding]
header:
  image: "/images/2020-03-17-compilation_cython/yancy-min-842ofHC6MaI-unsplash.jpg"
excerpt: "All the core concepts and syntax in a concise summary that comes with an Anki Deck to memorize all this stuff"
mathjax: "true"
---

You'll find below a summary of the most important key features of the Python programming language. Its clear syntax is illustrated trough simple examples. 

I've made [an other cheat sheet with a focus on the __Object Oriented__ part of Python and on __functionnal programming__](https://obrunet.github.io/pythonic%20ideas/python_cheat_sheet_2/). You can find all of this content in a dedicated [anki deck](https://github.com/obrunet/Memory_systems_-_Anki_decks/blob/master/01.My_own_decks/Programming_languages/Python/Python%203%20Cheat%20Sheet%201-2.apkg) to help you memorizing it.

Cards are composed of a simple challenge, then answer shows the code and its' result in [a dedicated jupyter notebook also available on github](https://github.com/obrunet/Memory_systems_-_Anki_decks/blob/master/01.My_own_decks/Programming_languages/Python/Python%203%20Cheat%20Sheet%201-2.ipynb).




## Design

what are the characteristics of the Python programming language ?
- interpreted
- high-level
- philosophy emphasizes on code readability (indentation)
- object-oriented
- dynamically-typed
- garbage-collected
- structured (procedural) 
- functional programming
- described as a "batteries included"

## Definitions  

__Explain what means "interpreted" ?__  
An interpreter is a computer program that directly executes instructions written in a programming or scripting language, without requiring them previously to have been compiled into a machine language program. 

__Explain what means "high-level" ?__  
A high-level programming language is a programming language with strong abstraction from the details of the computer.

__Explain what means "object-oriented" ?__    
Object Oriented programming (OOP) is a programming paradigm that relies on the concept of classes and objects. It is used to structure a software program into simple, reusable pieces of code blueprints (usually called classes), which are used to create individual instances of objects.

__Explain what means "dynamically-typed" ?__    
This means that the Python interpreter does type checking only as code runs, and the type of a variable is allowed to change over its lifetime.

__Explain what means "garbage-collected" ?__   
Garbage collection (GC) is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program, but is no longer referenced—also called garbage.

__Explain what means "structured (procedural)" ?__     
Structured programming is a programming paradigm aimed at improving the clarity, quality, and development time by making extensive use of the structured control flows and repetition, block structures.
Procedural programming is an otherparadigm based on the concept of the routine or subroutine(series of computational steps). 

__Explain what means "functional programming" ?__   
Functional programming is a paradigm where programs are constructed by applying and composing functions (rather than a sequence of imperative statements which update the running state of the program).

__Explain why Python is described as "batteries included" ?__    
This is due to its comprehensive standard library

## Basic Types

__Boolean__  
What are the values of a boolean variable ?


```python
True
False
```




    False



__Integers__  
Integer values - all kinds, also in binary or hex


```python
0
-192
0b010
0xF3
```




    243



__Strings__   
a basic string


```python
st = "One\tTwo\nThree"
st
```




    'One\tTwo\nThree'




```python
print(st)
```

    One	Two
    Three


a multiple line string


```python
st = """multiple 
lines
long
string"""

st
```




    'multiple \nlines\nlong\nstring'




```python
print(st)
```

    multiple 
    lines
    long
    string


a string with a single quote inside


```python
st = "I'm"
print(st)
```

    I'm


escape a single quote in a string


```python
st = 'I\'m'
print(st)
```

    I'm


## Advanced Types / Containers

__Lists__  
an empty list


```python
[]
```




    []



a list with various values  


```python
['x', 11, 8.9]
```




    ['x', 11, 8.9]



a list with a single element


```python
['st']
```




    ['st']



__Tuples__  
an empty tuple


```python
()
```




    ()



a tuple with various elements


```python
11, 'y', 7.4
```




    (11, 'y', 7.4)



a tuple with a single element


```python
('st')
```




    'st'



__Dictionnary__  
an empty dictionnary


```python
{}
```




    {}



a dictionnary with 2 keys


```python
{1: 'one', 'two': 2}
```




    {1: 'one', 'two': 2}



__Set__  
a set with various elements


```python
{1, 1, 2, 3}
```




    {1, 2, 3}



__Type conversion__  
convert a float to an integer


```python
int(1.2)
```




    1



convert an integer to a float


```python
float(1)
```




    1.0



convert an integer to a boolean of value True


```python
bool(1)
```




    True




```python
bool(2)
```




    True



convert an integer to a boolean of value False


```python
bool(0)
```




    False



convert an integer to a string


```python
str(123)
```




    '123'



convert a boolean to a string


```python
str(True)
```




    'True'



convert a set to a list


```python
list({1 ,2, 3})
```




    [1, 2, 3]



convert a string to a list


```python
list("adfs")
```




    ['a', 'd', 'f', 's']



convert a list to a string


```python
my_list = ['I', "'", 'l', 'u', 'v']
my_list
```




    ['I', "'", 'l', 'u', 'v']




```python
''.join(my_list)
```




    "I'luv"



__Iterables objects - properties__  

/ | List | Tuple | Set | Dict
---|---|---|---|---
symbol|[,]|(,)|{,}|{k: v,}
type|ordered|ordered|unordered|unordered
access|index|index|value|key
property|mutable|immutable|unique values|keys/values


---



## Assignements

multiple assignement of the same value to 3 different variables


```python
a = b = c = 0
a +=1
a, b
```




    (1, 0)



assign 2 different values to 2 differents variables in one line


```python
a, b = 1, 2
a, b
```




    (1, 2)



swap 2 variables' values


```python
a, b = 2, 8
a, b = b, a
a, b
```




    (8, 2)



list unpacking


```python
x, *y, z = (10, 'a', 'b', 20)
x, z
```




    (10, 20)



string unpacking


```python
i, j = 'ab'
i, j
```




    ('a', 'b')



get the last line of a log


```python
*content, last_line = open('/etc/fstab')
last_line
```




    '# UNCONFIGURED FSTAB FOR BASE SYSTEM\n'



decrement a variable


```python
a = 0
a -= 1
a
```




    -1



find the remainder of a division


```python
a = 16
a %= 5
a
```




    1



find the result of a division


```python
a = 16
a /= 5
a
```




    3.2



calculate the quotient of a division


```python
a = 16
a //= 5
a
```




    3



delete a variable


```python
a = 0
del a
a
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-c1e82bdf26d0> in <module>()
          1 a = 0
          2 del a
    ----> 3 a
    

    NameError: name 'a' is not defined


delete a user-defined objects, lists etc


```python
a, l = 1, [1, 2, 3]
del a, l
a, l
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-10-4d62949f80bf> in <module>
          1 a, l = 1, [1, 2, 3]
          2 del a, l
    ----> 3 a, l
    

    NameError: name 'a' is not defined


delete items within lists, dictionaries


```python
l = [1, 2, 3]
del l[1]
l
```




    [1, 3]



last value used by the interpreter


```python
10
```




    10




```python
_
```




    10



ignore a specific value


```python
a, _, b = (1, 2, 3)
_
```




    2



ignore specific values


```python
a, *_, b = (1, 2, 3, 4, 5)
_
```




    [2, 3, 4]



## Calculation / Operators  
"+   -   *   /   **   %   //   ~"

complementary


```python
~1
```




    -2



check if two objects are the same object:


```python
a = 'test'
b = a
a is b
```




    True



test if two objects that are equal, but not the same object:


```python
x = ["apple", "banana"]
y = ["apple", "banana"]
x is y
```




    False




```python
x is not y
```




    True




```python
x = y
x is y
```




    True



test if two objects are not the same object:


```python
a = 'test'
b = 'TEST'
a is not b
```




    True



test if an element is in a list


```python
'a' in ['c', 'b', 'a']
```




    True



test if an element is not in a list


```python
'z' not in ['c', 'b', 'a']
```




    True



the opposite of True


```python
not True
```




    False



Or condition with 2 booleans 


```python
True or False
```




    True



And condition with 2 booleans 


```python
True and True
```




    True


## Misc Operators

[Source w3schools](https://www.w3schools.com/python/python_operators.asp)

### Python Arithmetic Operators
![png](/images/2020-11-15-python_cheat_sheet_1/1_arithmetic.png)


### Python Assignment Operators
![png](/images/2020-11-15-python_cheat_sheet_1/2_assignment.png)


### Python Comparison Operators
![png](/images/2020-11-15-python_cheat_sheet_1/3_comparison.png)


### Python Logical Operators
![png](/images/2020-11-15-python_cheat_sheet_1/4_logical.png)


### Python Identity Operators
![png](/images/2020-11-15-python_cheat_sheet_1/5_identity.png)


### Python Membership Operators
![png](/images/2020-11-15-python_cheat_sheet_1/6_membership.png)


### Python Bitwise Operators
![png](/images/2020-11-15-python_cheat_sheet_1/7_bitwise.png)



```python
0b0011 & 0b1100
```




    0



or bitwise operator


```python
0b0011 | 0b1100
```




    15



Shift left by pushing zeros in from the right and let the leftmost bits fall off


```python
0b0001 << 1
```




    2



Inverts all the bits


```python
~ 0b0001
```




    -2



Sets each bit to 1 if only one of two bits is 1


```python
0b0001 ^ 0b0000
```




    1



test if 2 variables are equal


```python
a, b = 10, 10.0
a == b
```




    True



test if 2 variables are different


```python
a, b = 1, 2
a != b
```




    True




```python
0 == False
```




    True




```python
# everything not equals to 0 is True
0 != True
```




    True



__Python Operator Precedence__
From Python documentation on [operator precedence (Section 5.15)](http://docs.python.org/reference/expressions.html)

Highest precedence at top, lowest at bottom.
Operators in the same box evaluate left to right.

![png](/images/2020-11-15-python_cheat_sheet_1/9_precedence.png)

concatenate 2 lists


```python
['a', 'b'] + [1, 2]
```




    ['a', 'b', 1, 2]



concatenate strings


```python
'this' + 'is' 'hot'
```




    'thisishot'



repeat the same string three times


```python
'test_' * 3
```




    'test_test_test_'



repeat the same list three times


```python
[1, 2, 3] * 3
```




    [1, 2, 3, 1, 2, 3, 1, 2, 3]



loop on the indexes & values of a list


```python
lst = ['a', 'b', 'c']
for i, j in enumerate(lst):
  print(f'index {i} & value {j}')
```

    index 0 & value a
    index 1 & value b
    index 2 & value c


list integers from 0 to 10


```python
range(10)
```




    range(0, 10)




```python
list(range(10))
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



list integers from 2 to 5


```python
list(range(2, 6))
```




    [2, 3, 4, 5]



list integers with a step of 2


```python
list(range(2, 7, 2))
```




    [2, 4, 6]



## Containers Ops

get the min value of a list


```python
lst = [1, 2, 3]
min(lst)
```




    1



get the max value of a list


```python
lst = [1, 2, 3]
max(lst)
```




    3



get the sum of all values of a list


```python
lst = [1, 2, 3]
sum(lst)
```




    6



get the length of a list


```python
lst = [1, 2, 3, 4]
len(lst)
```




    4



test if an element is in a list


```python
lst = [1, 2, 3]
1 in lst
```




    True




```python
lst = [1, 2, 3]
4 in lst
```




    False



get a sorted list


```python
lst = [3, 1, 2, 3]
sorted(lst)
```




    [1, 2, 3, 3]



get a list of tuples from 2 list of the same lenght


```python
lst_1 = [1, 2, 3]
lst_2 = ['a', 'b', 'c']
list(zip(lst_1, lst_2))
```




    [(1, 'a'), (2, 'b'), (3, 'c')]



check if all the value of a list are True


```python
lst = [True, True, True]
all(lst)
```




    True




```python
lst = [False, True, True]
all(lst)
```




    False



check if at least one value of a list is True


```python
lst = [False, False, True]
any(lst)
```




    True




```python
lst = [False, False, False]
any(lst)
```




    False



reverse a list


```python
lst = [3, 1, 2, 3]
list(reversed(lst))
```




    [3, 2, 1, 3]




```python
lst[::-1]
```




    [3, 2, 1, 3]



repeat the same string three times


```python
s = "string__"
s * 3
```




    'string__string__string__'



concatenate two strings


```python
"string___" + " " "test"
```




    'string___ test'



get the index of an element


```python
lst = [3, 1, 2, 3]
lst.index(1)
```




    1




```python
lst = [3, 1, 2, 3]
lst.index(3)
```




    0



count the number of an element


```python
lst = [3, 1, 2, 3]
lst.count(1)
```




    1




```python
lst = [3, 1, 2, 3]
lst.count(3)
```




    2



## List Operators

add an element at the end of a list


```python
lst = [3, 1, 2, 3]
lst.append(0)
lst
```




    [3, 1, 2, 3, 0]



remove the 2nd element of a list


```python
lst = [3, 1, 2, 3]
lst.remove(1)
lst
```




    [3, 2, 3]



reverse the elements of a list in the same variable


```python
lst = [4, 1, 2, 3]
lst.reverse()
lst
```




    [3, 2, 1, 4]



sort the elements of a list in the same variable


```python
lst = [4, 1, 2, 3]
lst.sort()
lst
```




    [1, 2, 3, 4]




```python
lst = [4, 1, 2, 3]
sorted(lst)
```




    [1, 2, 3, 4]




```python
lst
```




    [4, 1, 2, 3]



add a list at the end of an other list


```python
lst = [4, 1, 2, 3]
lst.extend([0, 0])
lst
```




    [4, 1, 2, 3, 0, 0]



add an element at the beginning of a list


```python
lst = [4, 1, 2, 3]
lst.insert(0, 7)
lst
```




    [7, 4, 1, 2, 3]



add an element at the 2nd place of a list


```python
lst = [4, 1, 2, 3]
lst.insert(1, 7)
lst
```




    [4, 7, 1, 2, 3]



add an element at the end of a list (with insert or plus)


```python
lst = [4, 1, 2, 3]
lst.insert(-1, 7)
lst
```




    [4, 1, 2, 7, 3]




```python
[4, 1, 2, 3] + [7]
```




    [4, 1, 2, 3, 7]



retrieve the 2nd element of a list and delete it


```python
lst = [4, 7, 1, 2, 3]
lst.pop(1)
```




    7




```python
lst
```




    [4, 1, 2, 3]



## Dict Operators

get the value of a corresponding dict key


```python
d = {'a': 1, 'b': 2, 'c': 3}
d['a']
```




    1



list all keys of a dict


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.keys()
```




    dict_keys(['a', 'b', 'c'])



list all values of a dict


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.values()
```




    dict_values([1, 2, 3])



list tuples (k, v) of a dict 


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.items()
```




    dict_items([('a', 1), ('b', 2), ('c', 3)])



get the value of a corresponding dict key (with a default one)


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.get('a', 7)
```




    1




```python
d = {'a': 1, 'b': 2, 'c': 3}
d.get('z', 7)
```




    7



get the key of the min value in a dict


```python
d = {'a': 1, 'b': 2, 'c': 0}
min(d, key=d.get)
```




    'c'



removes the item that was last inserted into the dict & returns it


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.popitem()
```




    ('c', 3)




```python
d
```




    {'a': 1, 'b': 2}



removes & returns an item in the dict 


```python
d = {'a': 1, 'b': 2, 'c': 3}
d.pop('b')
```




    2




```python
d
```




    {'a': 1, 'c': 3}



## String Operators

convert a string in uppercase


```python
st = "abcdefgh"
st.upper()
```




    'ABCDEFGH'




```python
st
```




    'abcdefgh'



convert a string in lowercase


```python
st = "ABCDEFGH"
st.lower()
```




    'abcdefgh'



put a capital at the beginning of a string


```python
st = "this is a test"
st.capitalize()
```




    'This is a test'



replace a char by an other in a string


```python
st = "abcdefgh"
st.replace('e', 'Z')
```




    'abcdZfgh'




```python
st = "EabcdEfghE"
st.replace('E', 'Z')
```




    'ZabcdZfghZ'



remove the spaces at the beginning / the end of a string


```python
st = "  this is a test  "
st.strip(' ')
```




    'this is a test'



create a title from a string


```python
st = "this is a test"
st.title()
```




    'This Is A Test'



split lines in a string


```python
st = "this is a 1st line\nthis is a 2nd line"
print(st)
print(st.splitlines())
```

    this is a 1st line
    this is a 2nd line
    ['this is a 1st line', 'this is a 2nd line']


use an f_string with a float


```python
v = 3.14159
f'value of Pi: {v:.2f}'
```




    'value of Pi: 3.14'



test if a string contains only letters


```python
name = "Monica"
name.isalpha()
```




    True




```python
# contains whitespace
name = "Monica Geller"
name.isalpha()
```




    False




```python
# contains number
name = "Mo3nicaGell22er"
name.isalpha()
```




    False



test if a string contains only letters & numbers


```python
name = "Mo3nicaGell22er"
name.isalnum()
```




    True




```python
name = "Mo3nica Gell22er"
name.isalnum()
```




    False



## Loops

a simple for loop


```python
for i in range(0, 3):
  print(i**3)
```

    0
    1
    8


a simple while loop


```python
i = 0
while i < 4:
  print(i**3)
  i += 1
```

    0
    1
    8
    27


a while loop with conditions to break / continue


```python
i = 0
while i < nb:
  if cond1:
    break
  elif cond2:
    continue
  else:
    pass
  i += 1
```

get user input


```python
st = input("Enter your name: ")
print(f"You're name is {st}", "test", sep="---", end="___")
```

    Enter your name: Me
    You're name is Me---test___

## Comprehensions

a list comprehension to get even number with only a if (no elese) 


```python
[i for i in range(10) if i%2 == 0]
```




    [0, 2, 4, 6, 8]



a list comprehension to list even/odd number (with if and else cases)


```python
["Even" if i%2==0 else "Odd" for i in range(10)]
```




    ['Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd']



a dictionnary comprehension


```python
d = {1: 10, 2: 20, 3: 30}
{k**2: v+1 for (k, v) in d.items()}
```




    {1: 11, 4: 21, 9: 31}



reverse keys/values with a comprehension


```python
d = {1: 10, 2: 20, 3: 30}
{v: k for (k, v) in d.items()}
```




    {10: 1, 20: 2, 30: 3}



with a comprehension get a set of squared element from a list


```python
lst = [1, 2, 3, 3, 2, 1]
{i**2 for i in lst}
```




    {1, 4, 9}



## Case, switch statement

a switch statement with default value


```python
def zero():
    return "ZERO"
 
def one():
    return "ONE"
 
def two():
    return "TWO"
 
switcher = {
        0: zero,
        1: one,
        2: two
    }
 
 
def numbers_to_strings(argument):
    """Returns the func from switcher dic"""
    func = switcher.get(argument, "nothing")
    return func()
 
numbers_to_strings(1)
```




    'ONE'




```python
#changing the switch case
switcher[1]=two

numbers_to_strings(1)
```




    'TWO'




```python
switcher[0]()
```




    'ZERO'



## Exceptions

Python Built-in Exceptions


```python
print(dir(locals()['__builtins__']))
```

    ['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', '__IPYTHON__', '__build_class__', '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'display', 'divmod', 'dreload', 'enumerate', 'eval', 'exec', 'execfile', 'filter', 'float', 'format', 'frozenset', 'get_ipython', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'runfile', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']


how to handle exceptions


```python
try:
   # do something
except ValueError:
   # handle ValueError exception
except (TypeError, ZeroDivisionError):
   # handle multiple exceptions
   # TypeError and ZeroDivisionError
except:
   # handle all other exceptions
else:
  # if no exceptoin
finally:
  # for all cases
```

## Recursive functions

factorial recursive function


```python
def fact(n):
  """Calculates the factorial of n"""
  if n == 1:
    return 1
  else:
    return n * fact(n-1)


fact(4)
```




    24



cumulative sum with a recursive function


```python
def cum_sum(n):
  """Calculates the cumulative"""
  if n == 1:
    return 1
  nb = n
  temp = cum_sum(n-1)
  return nb + temp


cum_sum(4)
```




    10



## Lambda / map / filter

a simple lambda function with a tuple as input


```python
(lambda x, y: x + y)(2, 3)
```




    5



a lambda function that return the square of a value


```python
(lambda x: x**3)(2)
```




    8




```python
high_ord_func = lambda x, func: x + func(x)
high_ord_func
```




    <function __main__.<lambda>>




```python
high_ord_func(5, lambda x: x*x)
```




    30




```python
def sq(x):
  return x**2

print(map(sq, [1, 2, 3]))
```

    <map object at 0x7ff8e5483550>



```python
list(map(sq, [1, 2, 3]))
```




    [1, 4, 9]




```python
nb = (1, 2, 3, 4)
set((lambda x: x*x, nb))
```




    {(1, 2, 3, 4), <function __main__.<lambda>>}




```python
nb = (1, 2, 3, 4)
res = map(lambda x: x*x, nb)
set(res)
```




    {1, 4, 9, 16}




```python
a = list(range(10))
list(filter(lambda x: x%2 == 0, a))
```




    [0, 2, 4, 6, 8]



## PEP 8

[Source Real Python](https://realpython.com/python-pep8/)

![png](/images/2020-11-15-python_cheat_sheet_1/8_pep.png)
 

_single_leading_underscore  
This convention is used for declaring private variables, functions, methods and classes in a module. Anything with this convention are ignored in from module import *. 

single_trailing_underscore_  
This convention could be used for avoiding conflict with Python keywords or built-ins. You might not use it often.

\__\__double_leading_and_trailing_underscore\__\__    
This convention is used for special variables or methods (so-called “magic method”) such as__init__, \__\__len\__\__. These methods provides special syntactic features or does special things. For example, \__\__file\__\__ indicates the location of Python file, \__\__eq\__\__ is executed when a == b expression is excuted. 
A user of course can make custom special method, it is very rare case, but often might modify the some built-in special methods. (e.g. You should initialize the class with \__\__init\__\__ that will be executed at first when a instance of class is created.)

## Files Operation

open a file & print line one by one


```python
with open("file_path/dir", "w/r/a", encoding='utf8') as f:
  for line in f.readlines():
    print(line.split("car"), end=' ')
```

open a file & read lines


```python
with open("file_path/dir", "w/r/a", encoding='utf8') as f:
  data = f.read().splitlines()
```

open a file & write line one by one or all together


```python
with open("file_path/dir", "w/r/a", encoding='utf8') as f:
  ...
  f.write() # or
  f.writelines()
```

## Various Tricks

reverse a string


```python
"This is a string"[::-1]
```




    'gnirts a si sihT'



conditional assignement in one line (without 'else')


```python
x, y = 0, 10
if y == 10: x = 5
x 
```




    5



concatenate 2 dictionnaries


```python
i = {'a': 1, 'b': 2, 'c': 3}
j = {'d': 4, 'e': 5}
{**i, **j}
```




    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}



print the path of the 'os' module


```python
import os
print(os)
```

    <module 'os' from '/usr/lib/python3.7/os.py'>


conditional assignement in one line with 'else' case


```python
y = 10
x = 'a' if y == 10 else 'b'
x
```




    'a'




```python
y = 10
x = 'a' if y != 10 else 'b'
x
```




    'b'



compute the frequency of a list elements


```python
lst = ['a', 'b', 'b', 'c', 'c', 'c']
freq_dict = {}

for i in lst:
  if i not in freq_dict.keys():
    freq_dict[i] = 1
  freq_dict[i] += 1


freq_dict
```




    {'a': 2, 'b': 3, 'c': 4}



get the help of a module function


```python
from math import cosh
help(cosh)
```

    Help on built-in function cosh in module math:
    
    cosh(x, /)
        Return the hyperbolic cosine of x.
    


retrieve most frequent element of a list


```python
lst = ['a', 'b', 'b', 'c', 'c', 'c']
most_freq = max(set(lst), key=lst.count)
most_freq
```




    'c'




```python
def most_frequent(List):
    return max(set(List), key = List.count)
  
most_frequent(['a', 'b', 'b', 'c', 'c', 'c'])
```




    'c'



declare a Car & a Plane classes, create an instance of one of the two classes depending on an other value 


```python
class Car(object):
  def __init__(self, value):
    self.property = "car"
    self.value = value

class Plane(object):
  def __init__(self, param):
    self.property = "plane"
    self.param = param

y == 10
x = (Car if y == 10 else Plane)(22)
isinstance(x, Car)
```




    True




```python
z = (Car if y!= 10 else Plane)(33)
z.property, z.param
```




    ('plane', 33)


