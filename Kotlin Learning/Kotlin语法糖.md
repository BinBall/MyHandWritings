# Kotlin 语法糖

1.当函数体只有一行的时候，可以直接将函数体写在函数定义尾部，中间用等号连接**

```Kotlin
fun largerOne(a:Int,b:Int):Int= max(a,b)
```

2.所有定义了`setter`和`getter`方法的字段，在Kotlin中都可以通过赋值语法来直接操作

```kotlin
view.setOnClickListener { it.visibility = View.INVISIBLE }
```

3.?.操作符用合并为一个操作。只有当调用变量本身不为null时，方法调用才成立，否则整个表达式返回`null`。

```kotlin
fun getCountry(): String? {
    return person.company?.address?.country
}
```

4.字符串内嵌表达式（格式化输出）

```kotlin
//字符串内嵌表达式语法规则：
"Hello, ${obj.name}, nice to meet you!"

//当表达式中仅有一个变量时可以省略大括号：
"Hello, $name, nice to meet you!"
```

5.参数默认值

```kotlin
//使用默认值的参数不会被强制要求为其传递值，在没有传值时使用其默认值
fun printParams(num:Int,str:String="Hello"){
    println("num is $num, str is $str")
}
```

```kotlin
//若默认参数不是函数的最后一个参数，可以使用键值对传参
fun printParams(num:Int = 100,str:String="Hello"){
    println("num is $num, str is $str")
}
fun main(){
    printParams(str="123")
}
```

```kotlin
//通过为主构造函数添加参数默认值，可以在一定程度上替代次构造函数
class Student(val sno:String="",val grade:Int=0,name:String="",age:Int=0):Person(name,age){
}
```

6.快速使用Getter和Setter方法

```java
public class Book{//当在Kotlin中调用具有Setter和Getter方法的Java类时
	private int pages;
    public int getPages(){
        return pages;
    }
    public void getPages(int pages){
        this.pages=pages;
    }
}
```

```kotlin
val book=Book()
book.pages=500//Kotlin将代码自动转换成了调用Setter和Getter方法
val bookPages=book.pages
```

