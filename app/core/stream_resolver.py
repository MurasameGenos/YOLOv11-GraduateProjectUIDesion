import os
import re
import shutil  # 【新增】用于删除文件夹
from yt_dlp import YoutubeDL

BILI_RE = re.compile(r"(bilibili\.com/video/|b23\.tv/)")
CACHE_DIR = "bili_cache"

# 【新增】清理缓存的函数
def cleanup_cache():
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"[Stream Resolver] 已成功清理 B站 临时缓存目录")
        except Exception as e:
            print(f"[Stream Resolver] 清理缓存失败: {e}")

def resolve_stream_url(src: str) -> str:
    if not isinstance(src, str):
        return src
    if not src.startswith(("http://", "https://")):
        return src

    if BILI_RE.search(src):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        print(f"[Stream Resolver] 正在拉取 B站 视频到本地缓存，取决于网速，请稍候...")

        # 策略改变：不传直链了，直接将最高清的纯视频下载到本地
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            # 优先下载 mp4 格式的纯视频，体积小下载快
            "format": "bestvideo[ext=mp4]/bestvideo/best",
            # 存放到我们的缓存目录
            "outtmpl": f"{CACHE_DIR}/%(id)s_%(format_id)s.%(ext)s",
            "cookiefile": "cookies.txt",
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                # download=True 是关键，让它先下到本地
                info = ydl.extract_info(src, download=True)
                # 获取下载好的本地文件绝对路径
                filepath = ydl.prepare_filename(info)

                if os.path.exists(filepath):
                    print(f"[Stream Resolver] 视频已准备完毕: {filepath}")
                    # 将本地路径返回给 OpenCV
                    return filepath
        except Exception as e:
            print(f"[Stream Resolver] 缓存失败: {e}")

        return ""

    return src