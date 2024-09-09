#  安装与使用
## 克隆代码
```cmd
git clone https://github.com/xsyyiya/onnx2tflite.git

```
## 安装

```cmd
git checkout xsy
cd tools/parsetflite/
python setup.py install

```
可以使用以下命令生成whl文件，提供给其他人安装使用
```
sudo python setup.py sdist bdist_wheel
```
## 依赖
都是常见包，缺啥装啥

## 使用
安装后，可以在命令窗口直接使用parse-lite命令去解析tflite模型。

命令窗口键入:
```
parse-lite -h 
```
可查看其使用方式.



