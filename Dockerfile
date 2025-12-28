# 使用稳定的 Python 3.11 Slim 镜像 (Debian Bookworm)
FROM python:3.11-slim-bookworm

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖
# build-essential 用于编译一些 Python 扩展
# curl 用于健康检查
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到镜像中
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 启动命令
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
