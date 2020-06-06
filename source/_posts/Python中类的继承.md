---
title: Python中类的继承
date: 2020-04-11 22:41:59
categories: 学习
tags: Python

---

## 1. 类的继承
在编写类时，不是一定要从0开始，我们可以借助已有的类来完成工作。一个类继承另一个类时，被继承的类称为父类，继承的类称为子类，子类具有父类所有的属性和方法，并还能定义自己的属性和方法。
这里，我们继续沿用之前的Car()例子，定义一个ElectricCar()的新类，并继承Car()类的属性和方法。这样和我们事情情况相符，电动车具有普通汽车的大部分属性和功能，并且具有电动车独有的属性。<!--more-->
```python
class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__( make, model, year) #初始化父类的属性
        self.battery_size = 70 # 初始化子类的属性

    def describe_batteries(self): # 定义子类特有的方法
       '''打印关于battery的信息'''
    	print('This car has a ' + str(self.battery_size) + 'kmh battery.')
		
```
如上所示，定义了ElectricCar()类，并用```super().__init__(self)```初始化父类的属性，这里的```super()```是一个特殊的函数，专门用于类的继承，可理解为在父类和子类之间建立某种联系，使两者不再独立存在。子类的实例化和父类是一样的方法。

定义玩子类之后，我们需要为子类创建属性和方法。例如，```self.battery_size = 70 ```和 ```def describe_batteries(self) ``` battery_size和describe_batteries只有子类的实例可以调用。
### 重写父类的方法
对于父类已经定义的方法，只要子类的实例不符合，都可以进行修改。为此，可在子类中定义与父类方法同名的方法，这样，在实例化的时候便可实现对父类方法的更改。
假设我们在父类Car()中定义了一个属性：
```python 
class Car():
	'''snip'''
	
    def fill_gas_tank(self):
            print("This car's gas tank has been filled.")
```
该方法对于电动车来说毫无意义，因此我们在子类中重写它：
```python
class ElectricCar(Car):
	'''snip'''
	
	def fill_gas_tank(self): # 更改父类的方法，
    	print("This car doesn't need a tank.")
```
使用该方法，便可实现在继承父类的时候，对其无意义、不符合实际要求的方法进行修改。

### 将实例用作属性
在实际写程序的时候，一个类里面可能会添加很多的属性和方法，在这种情况下，我们可以把具有相同特点的属性和方法提取出来，单独用一个类来定义，这样在代码出错的时候，也方便debug。例如我们将ElectricCar()类中关于电瓶的属性和方法取出来，放到另一个名为Battery()的类中，并将一个Battery的实例用作ElecticCar()的属性。
```python
class Battery():
    def __init__(self, battery_size = 70):
        self.battery_size = battery_size

    def describe_batteries(self):
        '''打印关于battery的信息'''
        print('This car has a ' + str(self.battery_size) + 'kmh battery.')

    def get_range(self, model):
        if self.battery_size == 70:
            range = 240
        elif self.battery_size == 85:
            range = 280

        message = model + ' car can go approximately ' + str(range)
        message += ' miles on a full charge'
        print(message)
		
class ElectricCar(Car):
	'''snip'''
	
	def __init__(self, make, model, year):
    	super().__init__( make, model, year) #初始化父累的属性
		self.battery = Battery() # 将一个类做为另一个类的属性
```
在调用的时候可通过句点法访问Battery类。
```python
my_tesla = ElectricCar('tesla', 'model s', 2016)
my_tesla.battery.battery_size=85
my_tesla.battery.describe_batteries()
my_tesla.battery.get_range('model s')
```

## 2. 导入类
在实际工程中如果一个文件中定义了太多的类会显得很乱，为了让文件尽可能的简洁，我们可以将类存储为不同的模块，然后在主程序中调用。在python中只需在文件开头用```import ```指令便可实现对已定义模块的引用。
初次之外，我们还可以对python的[标准库](https://docs.python.org/3/library/index.html "标准库")进行应用，这些库有的是在安装python的时候就已经安装了，有些需要自行安装，例如numpy, matplotlib这类库可通过```pip```指令安装。导入库的时候，可一次导入一个模块```import numpy```，也可一次导入多个模块，模块之间用都好隔开```from car improt Car, ElectricCar```， 也可导入模块中的所有类```from module_name improt *```。

## 参考文献
[1] Matthes E. Python crash course: a hands-on, project-based introduction to programming[M]. No Starch Press, 2015.

