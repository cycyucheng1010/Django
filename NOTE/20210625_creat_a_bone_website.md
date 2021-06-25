# 20210625
## 檔案說明
* ```__init__.py``` 是一個空文件，指示 Python 將此目錄視為 Python 套件。
* ```settings.py``` 包含所有的網站設置。這是可以註冊所有創建的應用的地方，也是靜態文件，數據庫配置的地方，等等。
* ```urls.py``` 定義了網站url到view的映射。雖然這裡可以包含所有的url，但是更常見的做法是把應用相關的url包含在相關應用中，你可以在接下來的教程裡看到。
* ```wsgi.py``` 幫助Django應用和網絡服務器間的通訊。你可以把這個當作模板。
* ```manage.py``` 腳本可以創建應用，和資料庫通訊，啟動開發用網絡服務器。

## 創建catalog應用
* ```python3 manage.py start "catalog name"``` 將程式進行分類
## 註冊catalog應用

## 參考資料
* [Django 教學 2: 創建一個骨架網站](https://developer.mozilla.org/zh-TW/docs/Learn/Server-side/Django/skeleton_website)