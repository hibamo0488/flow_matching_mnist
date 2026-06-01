FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV http_proxy=${proxy}
ENV https_proxy=${proxy}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    build-essential \
    curl \
    git \
    vim \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy

## uvインストール. どの権限でも(システム全体で)uvが使えるように"/usr/local/bin"にコピー
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    rm /uv-installer.sh

# ## ホストのUID/GIDでユーザー作成
ARG UID
ARG GID
RUN groupadd -g ${GID} appgroup && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

WORKDIR /workspace

COPY . .

# # 非rootユーザで実行
RUN chown -R appuser:appgroup /workspace
USER appuser

CMD ["bash"]