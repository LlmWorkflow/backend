from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
import time

stablelm2 = Ollama(model="stablelm2:1.6b", base_url="http://10.200.1.45:11434")
qwen2 = Ollama(model="qwen2:7b", base_url="http://10.200.1.45:11434")
llama3 = Ollama(model="llama3:latest", base_url="http://10.200.1.45:11434")

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json  # 接收前端发送的 JSON 数据
    quest = data.get('text', '你好，请帮我用西班牙语翻译:"今天天气真好"')

    total_start_time = time.time()

    # 调用Qwen2模型
    start_time = time.time()
    res_qwen2 = qwen2.invoke(f"请将以下语句翻译为英文，请仅仅输出英文翻译, 问：{quest}，答：")
    end_time = time.time()
    qwen2_time = end_time - start_time

    # 调用StableLM2模型
    start_time = time.time()
    res_stablelm2 = stablelm2.invoke(f"Please translate the following statements into Spanish：Q：{res_qwen2}, A：")
    end_time = time.time()
    stablelm2_time = end_time - start_time

    # 调用Llama3模型
    start_time = time.time()
    res_llama3 = llama3.invoke(f"请将以下内容的答案部分采用西班牙语，其他部分采用中文：{res_stablelm2}")
    end_time = time.time()
    llama3_time = end_time - start_time

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # 返回 JSON 响应
    return jsonify({
        "qwen2_result": res_qwen2,
        "stablelm2_result": res_stablelm2,
        "llama3_result": res_llama3,
        "qwen2_time": qwen2_time,
        "stablelm2_time": stablelm2_time,
        "llama3_time": llama3_time,
        "total_time": total_time
    })

if __name__ == '__main__':
    app.run(debug=True)
