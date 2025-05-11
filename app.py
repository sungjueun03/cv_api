from flask import Flask, request, Response
from PIL import Image
from model import model_image
from flask_cors import CORS
from recommend_products import recommend_from_json
import json

app = Flask(__name__)
CORS(app)

@app.route('/model', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return Response(json.dumps({'error': '이미지가 없습니다'}, ensure_ascii=False), status=400)

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    try:
        result = model_image(image)

        skin_type = result.get("피부 타입", "중성")
        regions = {k: v for k, v in result.items() if k != "피부 타입"}
        formatted_result = {
            "regions": regions,
            "skin_type": skin_type
        }

        result_json = json.dumps(formatted_result, ensure_ascii=False)
        recommended_df = recommend_from_json(result_json)
        recommended_list = recommended_df.to_dict(orient='records')

        return Response(
            json.dumps({
                "cv_result": result,
                "recommended": recommended_list
            }, ensure_ascii=False, indent=2),
            content_type='application/json; charset=utf-8'
        )

    except Exception as e:
        return Response(json.dumps({'error': str(e)}, ensure_ascii=False), status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
