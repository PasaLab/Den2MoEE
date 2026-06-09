import tqdm
from pathlib import Path

import requests
from tqdm import tqdm

def download_file(url, save_path, desc=None):
    try:
        # 发起请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 获取文件总大小（字节）
        total_size = int(response.headers.get('content-length', 0))
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            desc=desc
        )

        # 写入文件
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉 keep-alive 块
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        print(f"\n文件已下载到: {save_path}")

    except Exception as e:
        print(f"下载失败: {e}")
        import pdb; pdb.set_trace()



path = Path("mtebdata/")
file_list = list(path.glob("**/*"))
file_list.sort()

for file_id, file in enumerate(file_list, 1):
    if file.is_dir():
        continue
    try:
        txt = open(file, "r", encoding="utf-8").read()
    except:
        print(f"{file} 读取失败")
        continue
    if txt.startswith("version"):
        file_path = f"evaluation/{file}"  # 确保路径正确包含evaluation目录
        url = f"https://media.githubusercontent.com/media/QwenLM/Qwen3-Embedding/refs/heads/main/{file_path}"
        download_file(
            url=url,
            save_path=file._str,
            desc=f"下载 {file_id}/{len(file_list)}-[{file.stem}]"
        )

