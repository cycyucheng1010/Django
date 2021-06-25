# 20210625
## 架設Django開發環境
* ```sudo apt install python3-pip``` 安裝pip
* ```sudo pip3 install virtualenvwrapper```安裝虛擬環境
* 編bashrc貼入以下資料:
```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV_ARGS=' -p /usr/bin/python3 '
export PROJECT_HOME=$HOME/Devel
source /usr/local/bin/virtualenvwrapper.sh
```
* ```pip3 install django```安裝django
* ```python3 -m django --version```linux/macos的django版本
## 常用指令
* ``` django-admin startproject " Projectname"```
* ``` python3 manage.py runserver```執行server

![image](https://user-images.githubusercontent.com/62127656/123414523-91307b00-d5e6-11eb-99d0-6450ccbb20eb.png)

## 參考資料
* [架設 Django 開發環境](https://developer.mozilla.org/zh-TW/docs/Learn/Server-side/Django/development_environment)
