import sys
import os
import ctypes
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from app.ui.main_window import MainWindow


def main():
    # 【避坑指南】告诉 Windows 这是一个独立的软件，而不是一个普通的 Python 脚本。
    # 只有加上这行代码，Windows 底部任务栏才会正确显示你自定义的图标！
    if os.name == 'nt':
        myappid = 'my_yolo_project.frontend.version.1'  # 这个字符串可以随便填，但必须有
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 开启高分屏支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载并设置全局图标
    icon_path = os.path.join(current_dir, "app_icon.ico")  # 如果你用的是 png，把后缀改成 .png
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    else:
        print(f"【提示】未找到图标文件: {icon_path}，将使用默认图标。")

    # 2. 加载 Win11 样式表
    qss_path = os.path.join(current_dir, "win11_dark.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            print("样式表加载成功！")

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()