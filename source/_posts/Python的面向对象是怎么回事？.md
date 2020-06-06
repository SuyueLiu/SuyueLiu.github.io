---
title: Python中类的创建和使用？
date: 2020-04-08 00:10:26
categories: 学习
tags: Python

---

## 1. 创建和使用类

一个类只提供定义一种对象的结构，如用class Dog()来记录小狗。我们都知道每个狗狗都有姓名和年龄，这是所有狗的通用属性。小狗还能蹲下和打滚，这种行为每个狗狗都会，在类的定义中称为方法。 这两项信息属于定义小狗实例所必须的，但是实际上在Dog()中不会包含任何特定的小狗。

## 2. Python对象实例化
python中的类设定了定义一个对象所需要的属性和方法，实例是指具有实际值的类的副本，例如Dog()中的一只具体的小狗，例如Tom，3岁，表示有一只3岁的小狗叫Tom。
可以把类比做一个表格，这个表格设定了所有要记录的东西，每记录一个/组数据就是一个实例。<!--more-->

## 3. Python中定义类
```python
class Dog():
    species = ‘mammal’
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def sit(self):
        print(self.name.title() + 'is now sitting')
    def roll_over(self):
        print(self.name.title() + 'rolled over')
```
在python3中用class关键字 + 类名称 来定义类。在python2中还需要指定类中继承的父类，python3中改为隐式默认值了。
类中的函数称为方法，如__init__, sit(), roll_over()都是方法。但是这里的__init__属于一个特殊的方法，每当你根据Dog类来创建一个实例时，程序都会自动运行这个方法给每个对象赋于name和age两个属性。 

在这里__init__(self, name, age)方法具有三个形参，self、name、age。其中形参self是必不可少的，且必须位于其余的形参之前。在创建Dog类的实例时，将自动传入实参slef。每个与类相关联的方法的调用都会自动传入实参self，他是一个指向实例本身的引用，每个实例都能够访问类的属性和方法。对于__init__方法中的形参name和age，在创建实例的时候必须传入相应的值。
注意这里self.name和self.age 定义的是name和age两个变量，和形参name和age不同。用self为前缀的变量都可被类中所有的方法使用。我们还可以通过类的所有实例来访问这些变量

类属性有别于实例属性。实例属性是每个特定实例所特有的属性、特征，如每一条狗的名字和年纪。类属性是所有属于这个类的实例所具有的属性，如这里的species = ‘mammal’，所有的小狗都属于mammal。

## 4. 创建类的实例
```python
my_dog = Dog('wills', 6)
your_dog = Dog('Tom', 3)
```
这两条代码分别创建了一条名为“wills”年龄6岁的小狗和名为“Tom”年龄为3岁的小狗，并把这两个实例分别存储到变量my_dog和your_dog里面，通过“句点法”我们便可访问实例的属性和调用类的方法。

```python
my_dog = Dog('wills', 6)
my_dog.sit()
print(my_dog.name + ' is my dog, ' + 'he is ' + str(my_dog.age) + ' years old.')

your_dog = Dog('Tom', 3)
your_dog.roll_over()
print('Your dog is ' + your_dog.name + ', he is ' + str(your_dog.age) + ' years old.')
```

## 5. 使用类和属性

### 5.1 给定属性的初始值/默认值

有时候我们需要给类里面的某个熟悉设定一个初始值或者默认值，可直接在__init__方法中指定相应的值，如果用这种方法的话，可无须在__init__中指定相应的形参。
我们重新定义一个Car()类：
```python
class Car():
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0 #如果在__init__内指定属性的初始值，无需包含为他提供初始值的形参

    def get_descriptive(self):
        long_name = str(self.year) + '  ' + self.make + '  ' + self.model
        return long_name.title()
   
   def read_odometer(self):
        print('This car has ' + str(self.odometer_reading) + ' miles on it.')
```

这里定义的Car()类除了self之外还包含三个形参：make, model, year, 我们想要设定汽车里程记初始值为0，但我们并未指定相应的形参。
下面创建一个实例，并调用方法get_descriptive()和read_odometer()
```python
my_car = Car('audi', 'a4', 2016)
print(my_car.get_descriptive())
my_car.read_odometer()
```

### 5.2 修改属性的值
有时候我们需要修改某个熟悉的值，这里提供三种方法：

#### 5.2.1 直接修改
	my_car.odometer_reading = 23
	my_car.read_odometer()

最简单的一种方法，通过实例访问属性并修改。但是这种方法无法对赋值的过程进行限定，比如没办法限制回调里程记。

#### 5.2.2 创建一个方法来更新属性的值
```python
def update_odometer(self, mileage):
    '''禁止他人将里程表的值回调'''
    if mileage >= self.odometer_reading:
        self.odometer_reading = mileage
    else:
        print('You cannot roll back an odometer')

my_car.update_odometer(560)
my_car.read_odometer()
```
在类里面定义一个新的方法update_odometer()，并设定一个形参mileage来接收新的实参，并且用if判断禁止回调里程记。

#### 5.2.3 定义一个新的方法对属性的值进行递增
这个方法其实和第二个类似，只是修改了更新的规则，可每次增加传入的miles的值。
```python 
def increase_odometer(self, miles):
    if miles >= 0:
        self.odometer_reading += miles
    else:
        print('You cannot roll back an odometer.')
```



