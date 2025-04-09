from flask import Flask
from flask_socketio import SocketIO, emit
from langchain_community.llms import Ollama

stablelm2 = Ollama(model="stablelm2:1.6b", base_url="http://10.200.1.44:11434")
qwen2 = Ollama(model="qwen2:0.5b", base_url="http://10.200.1.44:11434")
deepseek = Ollama(model="deepseek-r1:1.5b", base_url="http://10.200.1.44:11434")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('translate')
def handle_translate(data):

    def process_stage(stage, model, prompt):
        emit('stage_start', {'stage': stage})

        result = model.invoke(prompt)

        emit('stage_result', {
            'stage': stage,
            'result': result
        })
        return result

    try:
        # 分阶段处理
        res1 = process_stage('qwen2', qwen2, f"翻译为英文: {data['text']}")
        res2 = process_stage('stablelm2', stablelm2, f"Translate to Spanish: {res1},only export the translation")
        res3 = process_stage('deepseek', deepseek, f"请将以下内容的答案部分保留西班牙语，其他部分翻译为中文： {res2}")

        emit('complete', {
            'final_result': res3
        })

    except Exception as e:
        emit('error', {
            'message': f"处理失败: {str(e)}",
        })


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
