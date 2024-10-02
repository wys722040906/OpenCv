# Ubuntu文件系统

### 1./usr/local/第三方包默认安装

- 可执行文件:默认/usr/local/bin
- 库文件(.so动态.lib静态)：/usr/local/lib
  - /usr/local/lib/cmake/opencv4(cmake配置文件)
- 头文件：/usr/local/incldue/opencv4
- 文档文件：/usr/local/share/opencv4
- 放系统管理员使用的系统管理命令:/usr/sbin
- 指定安装路径

```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/opt/opencv ..
```

### 2./opt--第三方库，软件包安装目录

```
/opt/
  ├── example/
  │    ├── bin/
  │    ├── lib/
  │    ├── share/
  │    └── ...
  ├── another-software/
  │    ├── bin/
  │    ├── lib/
  │    ├── share/
  │    └── ...
  └── ...
```

- 创建符号链接

```
sudo ln -s /opt/example/bin/example /usr/local/bin/example
```

### 3 .生成第三方库

- main.cpp -->/bin

```
gcc hello.c -o hello
mv hello /usr/local/bin/
```

- my_lib.c

```
gcc -shared -o libmylib.so -fPIC mylib.c
mv libmylib.so /usr/local/lib/
```

- 调用

```
// main.c
#include <stdio.h>
void greet();
int main() {
    greet();
    return 0;
}
```

- 链接

```
gcc main.c -o main -L/usr/local/lib -lmylib
```

### 4.命令添加到'PATH'变量

- 临时

```
export PATH=$PATH:/path/to/directory
```

- 永久

```
echo 'export PATH=$PATH:/path/to/directory' >> ~/.bashrc
source ~/.bashrc
```

- 己方.sh直接运行

```
export PATH=$PATH:/usr/local/bin
myscript.sh
```

